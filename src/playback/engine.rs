use super::stream::{BufferedSource, StreamingAudio};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::Duration,
};

#[derive(Default)]
struct OutputSources {
    current: Option<BufferedSource>,
    // 交接后暂存旧音源，留给控制器线程释放。
    next: Option<BufferedSource>,
    armed: bool,
}

struct OutputControl {
    playing: AtomicBool,
    volume: AtomicU32,
    error: Mutex<Option<String>>,
    sources: Mutex<OutputSources>,
    switched: AtomicBool,
}

impl Default for OutputControl {
    fn default() -> Self {
        Self {
            playing: AtomicBool::new(false),
            volume: AtomicU32::new(1_f32.to_bits()),
            error: Mutex::new(None),
            sources: Mutex::default(),
            switched: AtomicBool::new(false),
        }
    }
}

/// 按歌曲采样率和声道数提交 f32 PCM，由系统共享输出适配设备格式。
#[derive(Default)]
pub(super) struct PlayerEngine {
    device: Option<(cpal::Stream, cpal::StreamConfig)>,
    audio: Option<StreamingAudio>,
    next_audio: Option<StreamingAudio>,
    next_frozen: bool,
    control: Arc<OutputControl>,
}

impl PlayerEngine {
    pub fn load(
        &mut self,
        audio: StreamingAudio,
        mut source: BufferedSource,
    ) -> Result<(), String> {
        self.stop();
        self.ensure_device(source.sample_rate, source.channels)?;
        source.playback_gain = 0.;
        self.control.sources.lock().unwrap().current = Some(source);
        self.audio = Some(audio);
        Ok(())
    }

    pub fn stage_next(&mut self, audio: StreamingAudio, mut source: BufferedSource) {
        let primed = source.prime();
        let mut sources = self.control.sources.lock().unwrap();
        sources.armed = primed
            && sources.current.as_ref().is_some_and(|current| {
                current.sample_rate == source.sample_rate && current.channels == source.channels
            });
        sources.next = Some(source);
        self.next_audio = Some(audio);
    }

    /// 改队列或定位前先冻结交接，并结算可能已由回调完成的切换。
    /// 常规轮询只检查原子标志，避免和音频回调争抢锁。
    pub fn settle_gapless(&mut self, freeze: bool) -> bool {
        if self.next_audio.is_none() || (!freeze && !self.control.switched.load(Ordering::Acquire))
        {
            return false;
        }
        let mut sources = self.control.sources.lock().unwrap();
        if self.control.switched.swap(false, Ordering::AcqRel) {
            let retired = sources.next.take();
            drop(sources);
            self.next_frozen = false;
            self.audio = self.next_audio.take();
            drop(retired);
            return true;
        }
        if freeze {
            self.next_frozen |= std::mem::take(&mut sources.armed);
        }
        false
    }

    pub fn rearm_next(&mut self) {
        if std::mem::take(&mut self.next_frozen) {
            self.control.sources.lock().unwrap().armed = true;
        }
    }

    /// 手动切歌及跨格式续播可以复用已经准备好的音源。
    pub fn take_next(&mut self) -> Option<(StreamingAudio, BufferedSource)> {
        let audio = self.next_audio.take()?;
        let mut sources = self.control.sources.lock().unwrap();
        sources.armed = false;
        self.next_frozen = false;
        let source = sources.next.take().unwrap();
        Some((audio, source))
    }

    pub fn audio(&self) -> Option<&StreamingAudio> {
        self.audio.as_ref()
    }

    fn ensure_device(&mut self, sample_rate: u32, channels: u16) -> Result<(), String> {
        if self.device.as_ref().is_some_and(|(_, config)| {
            config.sample_rate == sample_rate && config.channels == channels
        }) {
            return Ok(());
        }
        self.device = None;
        *self.control.error.lock().unwrap() = None;
        let device = cpal::default_host()
            .default_output_device()
            .ok_or("没有可用的音频输出设备")?;
        let stream_config = cpal::StreamConfig {
            channels,
            sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };
        // ponytail: 依赖共享后端适配格式；其他后端拒绝时报告错误，支持它们时再加适配。
        let control = self.control.clone();
        let errors = self.control.clone();
        let stream = device
            .build_output_stream(
                &stream_config,
                move |buffer: &mut [f32], _| write_output(buffer, &control),
                move |error| {
                    errors.playing.store(false, Ordering::Release);
                    *errors.error.lock().unwrap() = Some(format!("音频输出失败：{error}"));
                },
                None,
            )
            .map_err(|error| format!("无法创建音频输出流：{error}"))?;
        // 保持流运行，暂停淡出结束后只写静音，不消费 PCM。
        stream
            .play()
            .map_err(|error| format!("无法启动音频输出：{error}"))?;
        self.device = Some((stream, stream_config));
        Ok(())
    }

    pub fn stop(&mut self) {
        self.pause();
        let sources = std::mem::take(&mut *self.control.sources.lock().unwrap());
        self.control.switched.store(false, Ordering::Release);
        self.next_frozen = false;
        self.next_audio = None;
        self.audio = None;
        drop(sources);
        // 正常停止保留输出设备；设备报错时释放，下次播放重新打开。
        let device_failed = self.control.error.lock().unwrap().is_some();
        if device_failed {
            self.device = None;
            *self.control.error.lock().unwrap() = None;
        }
    }

    pub fn has_source(&self) -> bool {
        self.audio.is_some()
    }

    pub fn finished(&self) -> bool {
        self.audio.as_ref().is_some_and(StreamingAudio::finished)
    }

    pub fn buffering(&self) -> bool {
        self.audio.as_ref().is_some_and(StreamingAudio::buffering)
    }

    pub fn download_complete(&self) -> bool {
        self.audio
            .as_ref()
            .is_some_and(StreamingAudio::download_complete)
    }

    pub fn download_failed(&self) -> bool {
        self.audio
            .as_ref()
            .is_some_and(StreamingAudio::download_failed)
    }

    pub fn stalled(&self, timeout: Duration) -> bool {
        self.audio
            .as_ref()
            .is_some_and(|audio| audio.stalled(timeout))
    }

    /// 播放地址已使用的时长；没有音源时为 None。
    pub fn source_age(&self) -> Option<Duration> {
        self.audio.as_ref().map(StreamingAudio::source_age)
    }

    pub fn pause(&self) {
        self.control.playing.store(false, Ordering::Release);
    }

    pub fn resume(&self) -> bool {
        let playing = self.has_source() && !self.finished() && self.error().is_none();
        self.control.playing.store(playing, Ordering::Release);
        playing
    }

    /// 新音源和自动重播直接起播；只有用户播放/暂停操作使用渐变。
    pub fn start(&self) -> bool {
        if let Some(source) = self.control.sources.lock().unwrap().current.as_mut() {
            source.playback_gain = 1.;
            // 与回调共用音源锁，不能在设置增益后先执行一次暂停淡出。
            return self.resume();
        }
        false
    }

    pub fn position(&self) -> Duration {
        self.audio
            .as_ref()
            .map_or(Duration::ZERO, StreamingAudio::position)
    }

    pub fn seek_to(&self, position: Duration) -> bool {
        if let Some(audio) = &self.audio {
            audio.seek(position);
            true
        } else {
            false
        }
    }

    pub fn error(&self) -> Option<String> {
        self.control
            .error
            .lock()
            .unwrap()
            .clone()
            .or_else(|| self.audio.as_ref().and_then(StreamingAudio::error))
    }

    pub fn volume(&self) -> f32 {
        f32::from_bits(self.control.volume.load(Ordering::Acquire))
    }

    pub fn set_volume(&mut self, volume: f32) -> bool {
        if !volume.is_finite() {
            return false;
        }
        self.control
            .volume
            .store(volume.clamp(0., 1.).to_bits(), Ordering::Release);
        true
    }
}

/// 回调只尝试获取音源锁，网络欠载、暂停或无下一首时输出静音。
fn write_output(buffer: &mut [f32], control: &OutputControl) {
    let Ok(mut sources) = control.sources.try_lock() else {
        buffer.fill(0.);
        return;
    };
    let OutputSources {
        current,
        next,
        armed,
    } = &mut *sources;
    let Some(source) = current.as_mut() else {
        buffer.fill(0.);
        return;
    };
    let playing = control.playing.load(Ordering::Acquire);
    let volume = f32::from_bits(control.volume.load(Ordering::Acquire));
    let end = source.write_playback(buffer, volume, playing);
    if end < buffer.len()
        && playing
        && *armed
        && let Some(successor) = next.as_mut()
    {
        successor.playback_gain = source.playback_gain;
        successor.write_playback(&mut buffer[end..], volume, playing);
        std::mem::swap(current, next);
        *armed = false;
        control.switched.store(true, Ordering::Release);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::stream::tests::short_wav;

    pub(in crate::playback) async fn play_short_wav(samples: &[i16]) -> PlayerEngine {
        let (audio, source) = short_wav(samples, 48_000, 2).await;
        let mut engine = PlayerEngine::default();
        engine.control.sources.lock().unwrap().current = Some(source);
        engine.audio = Some(audio);
        assert!(engine.start());
        engine
    }

    pub(in crate::playback) fn render_test_output(engine: &PlayerEngine, buffer: &mut [f32]) {
        write_output(buffer, &engine.control);
    }

    #[tokio::test]
    async fn output_joins_decoded_tracks_without_padding_or_losing_frames() {
        let samples = [1024, -2048, 3072, -4096, 0, 5120, -6144, 7168, 8192, -9216];
        // 分别让接缝落在 CPAL 缓冲区内部和缓冲区边缘。
        for first_buffer_size in [6, 4] {
            let mut engine = play_short_wav(&samples[..4]).await;
            let (next, output) = short_wav(&samples[4..], 48_000, 2).await;
            engine.stage_next(next, output);
            engine.set_volume(0.5);

            let mut actual = [f32::NAN; 10];
            let (first, second) = actual.split_at_mut(first_buffer_size);
            write_output(first, &engine.control);
            assert_eq!(
                engine.control.switched.load(Ordering::Acquire),
                first_buffer_size == 6
            );
            write_output(second, &engine.control);

            assert_eq!(actual, samples.map(|sample| f32::from(sample) / 65536.));
            assert!(engine.settle_gapless(false));
            assert!(!engine.settle_gapless(false));
            assert!(engine.position() >= Duration::from_micros(62));
            assert!(!engine.finished());

            let mut end = [1.; 2];
            write_output(&mut end, &engine.control);
            assert_eq!(end, [0.; 2]);
            assert!(engine.finished());
        }
    }

    #[tokio::test]
    async fn output_keeps_mismatched_or_frozen_successor_for_normal_loading() {
        for (rate, channels, freeze) in [(44_100, 2, false), (48_000, 1, false), (48_000, 2, true)]
        {
            let mut engine = play_short_wav(&[8192, -8192]).await;
            let (next, output) = short_wav(&[16384, -16384], rate, channels).await;
            engine.stage_next(next, output);
            if freeze {
                assert!(!engine.settle_gapless(true));
            }

            let mut frames = [1.; 4];
            write_output(&mut frames, &engine.control);
            assert_eq!(frames, [0.25, -0.25, 0., 0.]);
            assert!(engine.finished());
            assert!(!engine.settle_gapless(false));

            let (next, mut source) = engine.take_next().unwrap();
            assert_eq!(next.position(), Duration::ZERO);
            source.write(&mut frames[..channels as usize], 1.);
            assert_eq!(frames[0], 0.5);
        }
    }

    #[tokio::test]
    async fn output_does_not_consume_the_successor_while_paused() {
        let mut engine = play_short_wav(&[8192, -8192]).await;
        let (next, output) = short_wav(&[16384, -16384], 48_000, 2).await;
        engine.stage_next(next, output);
        engine.pause();
        let mut frames = [1.; 4];
        write_output(&mut frames, &engine.control);
        assert_eq!(frames[2..], [0.; 2]);
        assert!(!engine.settle_gapless(true));
        let (next, _) = engine.take_next().unwrap();
        assert_eq!(next.position(), Duration::ZERO);
    }

    #[test]
    fn output_silences_missing_paused_or_busy_source_without_blocking() {
        let control = OutputControl::default();
        let mut buffer = [1_f32; 4];
        write_output(&mut buffer, &control);
        assert_eq!(buffer, [0.; 4]);
        control.playing.store(true, Ordering::Release);
        buffer.fill(1.);
        write_output(&mut buffer, &control);
        assert_eq!(buffer, [0.; 4]);
        let _busy = control.sources.lock().unwrap();
        buffer.fill(1.);
        write_output(&mut buffer, &control);
        assert_eq!(buffer, [0.; 4]);
    }

    #[test]
    fn stop_clears_failed_device_for_retry_and_retains_volume() {
        let mut engine = PlayerEngine::default();
        engine.set_volume(0.4);
        *engine.control.error.lock().unwrap() = Some("设备已断开".into());
        assert!(engine.error().is_some());
        engine.stop();
        assert!(engine.error().is_none());
        assert!(engine.device.is_none());
        assert!(!engine.has_source());
        assert!(!engine.resume());
        assert_eq!(engine.volume(), 0.4);
    }

    #[test]
    fn volume_is_retained_without_a_device_and_ignores_non_finite_values() {
        let mut engine = PlayerEngine::default();
        assert_eq!(engine.volume(), 1.);
        assert!(engine.set_volume(0.4));
        assert_eq!(engine.volume(), 0.4);
        assert!(engine.set_volume(2.));
        assert_eq!(engine.volume(), 1.);
        assert!(engine.set_volume(-1.));
        assert_eq!(engine.volume(), 0.);
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(!engine.set_volume(invalid));
            assert_eq!(engine.volume(), 0.);
        }
        engine.stop();
        assert_eq!(engine.volume(), 0.);
    }
}
