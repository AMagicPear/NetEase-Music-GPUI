use super::stream::{BufferedSource, StreamingAudio};
use arc_swap::ArcSwap;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use smallvec::SmallVec;
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    },
    time::Duration,
};

// The low bits identify the current track. Updating the epoch when freezing
// prevents a callback with an old queue snapshot from committing after rearm.
const TRACK_MASK: u64 = u32::MAX as u64;
const EPOCH_STEP: u64 = 1 << 32;
const FROZEN: u64 = 1 << 63;

struct OutputSlot {
    id: u32,
    format: (u32, u16),
    ready: bool,
    source: Mutex<Option<BufferedSource>>,
}

struct Track {
    audio: StreamingAudio,
    slot: Arc<OutputSlot>,
}

#[derive(Default)]
struct OutputControl {
    playing: AtomicBool,
    start_at_full_gain: AtomicBool,
    error: Mutex<Option<String>>,
    // Volume defaults to zero bits, so construction explicitly initializes it.
    volume: AtomicU32,
    queue: ArcSwap<SmallVec<[Arc<OutputSlot>; 3]>>,
    gate: AtomicU64,
}

#[derive(Default)]
struct CallbackOutput {
    current: Option<(u32, BufferedSource)>,
}

pub(super) struct PlayerEngine {
    device: Option<(cpal::Stream, cpal::StreamConfig)>,
    current: Option<Track>,
    next: VecDeque<Track>,
    track_id: u32,
    control: Arc<OutputControl>,
    #[cfg(test)]
    output: std::cell::RefCell<CallbackOutput>,
}

impl Default for PlayerEngine {
    fn default() -> Self {
        Self {
            device: None,
            current: None,
            next: VecDeque::new(),
            track_id: 0,
            control: Arc::new(OutputControl {
                volume: AtomicU32::new(1_f32.to_bits()),
                ..Default::default()
            }),
            #[cfg(test)]
            output: Default::default(),
        }
    }
}

impl PlayerEngine {
    pub fn load(&mut self, audio: StreamingAudio, source: BufferedSource) -> Result<(), String> {
        self.stop();
        self.ensure_device(source.sample_rate, source.channels)?;
        self.install(audio, source);
        Ok(())
    }

    fn track(&mut self, audio: StreamingAudio, source: BufferedSource, ready: bool) -> Track {
        self.track_id = self.track_id.wrapping_add(1).max(1);
        Track {
            audio,
            slot: Arc::new(OutputSlot {
                id: self.track_id,
                format: (source.sample_rate, source.channels),
                ready,
                source: Mutex::new(Some(source)),
            }),
        }
    }

    fn install(&mut self, audio: StreamingAudio, mut source: BufferedSource) {
        source.playback_gain = 0.;
        self.current = Some(self.track(audio, source, true));
        self.publish_queue();
        self.control
            .gate
            .store(u64::from(self.track_id), Ordering::Release);
    }

    fn publish_queue(&self) {
        let slots = self
            .current
            .iter()
            .chain(&self.next)
            .map(|track| track.slot.clone());
        self.control.queue.store(Arc::new(slots.collect()));
    }

    pub fn stage_next(&mut self, audio: StreamingAudio, mut source: BufferedSource) {
        let ready = source.prime();
        let track = self.track(audio, source, ready);
        self.next.push_back(track);
        self.publish_queue();
    }

    fn freeze(&self) -> u64 {
        let mut gate = self.control.gate.load(Ordering::Acquire);
        loop {
            if gate & FROZEN != 0 {
                return gate;
            }
            let frozen = gate.wrapping_add(EPOCH_STEP) | FROZEN;
            match self.control.gate.compare_exchange_weak(
                gate,
                frozen,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return frozen,
                Err(actual) => gate = actual,
            }
        }
    }

    /// Freeze and observe the same atomic state used by the callback to claim a successor.
    pub fn settle_gapless(&mut self, freeze: bool) -> usize {
        let gate = if freeze {
            self.freeze()
        } else {
            self.control.gate.load(Ordering::Acquire)
        };
        let id = gate as u32;
        let Some(index) = self.next.iter().position(|track| track.slot.id == id) else {
            return 0;
        };
        let count = index + 1;
        for _ in 0..count {
            self.current = self.next.pop_front();
        }
        count
    }

    pub fn rearm_next(&self) {
        self.control.gate.fetch_and(!FROZEN, Ordering::Release);
    }

    pub fn clear_next(&mut self) {
        self.freeze();
        self.next.clear();
        self.publish_queue();
    }

    /// A frozen, unclaimed successor can be reused by ordinary loading.
    pub fn take_next(&mut self) -> Option<(StreamingAudio, BufferedSource)> {
        let claimed = self.freeze() as u32;
        if self
            .current
            .as_ref()
            .is_none_or(|track| track.slot.id != claimed)
        {
            return None;
        }
        let track = self.next.pop_front()?;
        self.publish_queue();
        let source = track.slot.source.lock().unwrap().take()?;
        Some((track.audio, source))
    }

    pub fn staged_download_complete(&self) -> bool {
        self.next
            .back()
            .is_some_and(|track| track.audio.download_complete())
    }

    pub fn audio(&self) -> Option<&StreamingAudio> {
        self.current.as_ref().map(|track| &track.audio)
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
        let open = || build_output::<f32>(&device, &stream_config, self.control.clone());
        #[cfg(target_os = "windows")]
        let stream = open().or_else(|float_error| {
            // Some WASAPI drivers reject WAVEFORMATEXTENSIBLE float but accept
            // standard PCM for the same sample rate and channel count.
            log::warn!("WASAPI f32 初始化失败，尝试 16-bit PCM：{float_error}");
            *self.control.error.lock().unwrap() = None;
            build_output::<i16>(&device, &stream_config, self.control.clone())
                .map_err(|pcm_error| format!("{float_error}；16-bit PCM：{pcm_error}"))
        })?;
        #[cfg(not(target_os = "windows"))]
        let stream = open()?;
        self.device = Some((stream, stream_config));
        Ok(())
    }

    pub fn stop(&mut self) {
        self.pause();
        self.freeze();
        self.control.gate.store(0, Ordering::Release);
        self.control
            .start_at_full_gain
            .store(false, Ordering::Release);
        self.next.clear();
        self.current = None;
        self.publish_queue();
        // 正常停止保留输出设备；设备报错时释放，下次播放重新打开。
        let device_failed = self.control.error.lock().unwrap().is_some();
        if device_failed {
            self.device = None;
            *self.control.error.lock().unwrap() = None;
        }
    }

    pub fn has_source(&self) -> bool {
        self.current.is_some()
    }

    pub fn finished(&self) -> bool {
        self.audio().is_some_and(StreamingAudio::finished)
    }

    pub fn buffering(&self) -> bool {
        self.audio().is_some_and(StreamingAudio::buffering)
    }

    pub fn download_complete(&self) -> bool {
        self.audio().is_some_and(StreamingAudio::download_complete)
    }

    pub fn download_failed(&self) -> bool {
        self.audio().is_some_and(StreamingAudio::download_failed)
    }

    pub fn stalled(&self, timeout: Duration) -> bool {
        self.audio().is_some_and(|audio| audio.stalled(timeout))
    }

    /// 播放地址已使用的时长；没有音源时为 None。
    pub fn source_age(&self) -> Option<Duration> {
        self.audio().map(StreamingAudio::source_age)
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
        self.control
            .start_at_full_gain
            .store(true, Ordering::Release);
        self.resume()
    }

    pub fn position(&self) -> Duration {
        self.audio()
            .map_or(Duration::ZERO, StreamingAudio::position)
    }

    pub fn seek_to(&self, position: Duration) -> bool {
        if let Some(audio) = self.audio() {
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
            .or_else(|| self.audio().and_then(StreamingAudio::error))
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

trait OutputSample: cpal::SizedSample {
    fn render(
        output: &mut CallbackOutput,
        buffer: &mut [Self],
        control: &OutputControl,
        channels: usize,
    );
}

impl OutputSample for f32 {
    fn render(output: &mut CallbackOutput, buffer: &mut [Self], control: &OutputControl, _: usize) {
        output.write(buffer, control);
    }
}

#[cfg(any(target_os = "windows", test))]
impl OutputSample for i16 {
    fn render(
        output: &mut CallbackOutput,
        buffer: &mut [Self],
        control: &OutputControl,
        channels: usize,
    ) {
        use cpal::Sample;
        let mut scratch = [0.; 1024];
        if !(1..=scratch.len()).contains(&channels) {
            buffer.fill(0);
            return;
        }
        let chunk_size = channels * (scratch.len() / channels);
        for chunk in buffer.chunks_mut(chunk_size) {
            let pcm = &mut scratch[..chunk.len()];
            output.write(pcm, control);
            for (sample, &value) in chunk.iter_mut().zip(pcm.iter()) {
                *sample = i16::from_sample(value);
            }
        }
    }
}

fn build_output<T: OutputSample>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    control: Arc<OutputControl>,
) -> Result<cpal::Stream, String> {
    let mut output = CallbackOutput::default();
    let channels = usize::from(config.channels);
    let errors = control.clone();
    let stream = device
        .build_output_stream(
            config,
            move |buffer: &mut [T], _| T::render(&mut output, buffer, &control, channels),
            move |error| {
                errors.playing.store(false, Ordering::Release);
                *errors.error.lock().unwrap() = Some(format!("音频输出失败：{error}"));
            },
            None,
        )
        .map_err(|error| format!("无法创建音频输出流：{error}"))?;
    stream
        .play()
        .map_err(|error| format!("无法启动音频输出：{error}"))?;
    Ok(stream)
}

impl CallbackOutput {
    fn write(&mut self, mut buffer: &mut [f32], control: &OutputControl) {
        let current = control.gate.load(Ordering::Acquire) as u32;
        if self.current.as_ref().map(|(id, _)| *id) != Some(current) {
            let queue = control.queue.load();
            self.current = queue
                .iter()
                .find(|slot| slot.id == current)
                .and_then(|slot| {
                    slot.source
                        .try_lock()
                        .ok()?
                        .take()
                        .map(|source| (slot.id, source))
                });
        }
        let playing = control.playing.load(Ordering::Acquire);
        let volume = f32::from_bits(control.volume.load(Ordering::Acquire));
        if playing
            && control.start_at_full_gain.swap(false, Ordering::AcqRel)
            && let Some((_, source)) = &mut self.current
        {
            source.playback_gain = 1.;
        }
        while let Some((current, source)) = &mut self.current {
            let end = source.write_playback(buffer, volume, playing);
            if end == buffer.len() {
                return;
            }
            if !playing {
                source.confirm_end();
                return;
            }
            // Read the gate before the immutable queue. A freeze invalidates
            // this claim even if the controller rearms before the CAS.
            let gate = control.gate.load(Ordering::Acquire);
            if gate & FROZEN != 0 || gate as u32 != *current {
                source.confirm_end();
                return;
            }
            let queue = control.queue.load();
            let mut after_current = queue.iter().skip_while(|slot| slot.id != *current).skip(1);
            let Some(next) = after_current
                .next()
                .filter(|slot| slot.ready && slot.format == (source.sample_rate, source.channels))
            else {
                source.confirm_end();
                return;
            };
            let claim = (gate & !TRACK_MASK) | u64::from(next.id);
            if control
                .gate
                .compare_exchange(gate, claim, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                source.confirm_end();
                return;
            }
            // Only the successful claimant can take this source. The controller
            // takes unclaimed sources only after freezing the gate.
            let Some(mut successor) = next.source.try_lock().ok().and_then(|mut item| item.take())
            else {
                return;
            };
            successor.playback_gain = source.playback_gain;
            self.current = Some((next.id, successor));
            buffer = &mut buffer[end..];
        }
        buffer.fill(0.);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::stream::tests::short_wav;

    pub(in crate::playback) async fn play_short_wav(samples: &[i16]) -> PlayerEngine {
        let (audio, source) = short_wav(samples, 48_000, 2).await;
        let mut engine = PlayerEngine::default();
        engine.install(audio, source);
        assert!(engine.start());
        engine
    }

    pub(in crate::playback) fn render_test_output(engine: &PlayerEngine, buffer: &mut [f32]) {
        engine.output.borrow_mut().write(buffer, &engine.control);
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
            render_test_output(&engine, first);
            assert_eq!(
                engine.control.gate.load(Ordering::Acquire) as u32
                    != engine.current.as_ref().unwrap().slot.id,
                first_buffer_size == 6
            );
            render_test_output(&engine, second);

            assert_eq!(actual, samples.map(|sample| f32::from(sample) / 65536.));
            assert_eq!(engine.settle_gapless(false), 1);
            assert_eq!(engine.settle_gapless(false), 0);
            assert!(engine.position() >= Duration::from_micros(62));
            assert!(!engine.finished());

            let mut end = [1.; 2];
            render_test_output(&engine, &mut end);
            assert_eq!(end, [0.; 2]);
            assert!(engine.finished());
        }
    }

    #[tokio::test]
    async fn output_consumes_two_successors_in_one_callback() {
        let mut engine = play_short_wav(&[8192, -8192]).await;
        for sample in [4096, 2048] {
            let (audio, source) = short_wav(&[sample, -sample], 48_000, 2).await;
            engine.stage_next(audio, source);
        }
        let mut frames = [f32::NAN; 6];
        render_test_output(&engine, &mut frames);
        assert_eq!(frames, [0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625]);
        assert_eq!(engine.settle_gapless(false), 2);
        assert_eq!(engine.settle_gapless(false), 0);
    }

    #[tokio::test]
    async fn integer_output_preserves_the_track_boundary() {
        let mut engine = play_short_wav(&[8192, -8192]).await;
        let (audio, source) = short_wav(&[4096, -4096], 48_000, 2).await;
        engine.stage_next(audio, source);
        let mut buffer = [0_i16; 4];
        i16::render(
            &mut engine.output.borrow_mut(),
            &mut buffer,
            &engine.control,
            2,
        );
        assert_eq!(buffer, [8192, -8192, 4096, -4096]);
    }

    #[tokio::test]
    async fn callback_continues_current_audio_during_queue_updates() {
        let mut engine = play_short_wav(&[8192, -8192, 4096, -4096]).await;
        let (audio, source) = short_wav(&[2048, -2048], 48_000, 2).await;
        engine.stage_next(audio, source);

        let queued = engine.next[0].slot.clone();
        let pending_slot = queued.source.lock().unwrap();
        let mut frames = [f32::NAN; 4];
        render_test_output(&engine, &mut frames);
        assert_eq!(frames, [0.25, -0.25, 0.125, -0.125]);
        drop(pending_slot);
        assert_eq!(engine.settle_gapless(true), 0);
        engine.clear_next();
        engine.rearm_next();
        render_test_output(&engine, &mut frames);
        assert_eq!(frames, [0.; 4]);
        assert_eq!(engine.settle_gapless(false), 0);
    }

    #[tokio::test]
    async fn output_keeps_mismatched_or_frozen_successor_for_normal_loading() {
        for (rate, channels, freeze) in [(44_100, 2, false), (48_000, 1, false), (48_000, 2, true)]
        {
            let mut engine = play_short_wav(&[8192, -8192]).await;
            let (next, output) = short_wav(&[16384, -16384], rate, channels).await;
            engine.stage_next(next, output);
            if freeze {
                assert_eq!(engine.settle_gapless(true), 0);
            }

            let mut frames = [1.; 4];
            render_test_output(&engine, &mut frames);
            assert_eq!(frames, [0.25, -0.25, 0., 0.]);
            assert!(engine.finished());
            assert_eq!(engine.settle_gapless(false), 0);

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
        render_test_output(&engine, &mut frames);
        assert_eq!(frames[2..], [0.; 2]);
        assert_eq!(engine.settle_gapless(true), 0);
        let (next, _) = engine.take_next().unwrap();
        assert_eq!(next.position(), Duration::ZERO);
    }

    #[test]
    fn output_silences_missing_source() {
        let engine = PlayerEngine::default();
        let mut buffer = [1_f32; 4];
        render_test_output(&engine, &mut buffer);
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
