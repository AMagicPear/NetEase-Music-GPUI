use super::StreamControl;
use std::{
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

pub(super) const BUFFER_SECONDS: f64 = 0.5;
const FADE_SECONDS: f32 = 0.08;

pub(super) struct Chunk {
    pub samples: Vec<f32>,
    pub position: Duration,
    pub generation: u64,
}

#[derive(Default)]
pub(super) struct PcmData {
    pub chunks: VecDeque<Chunk>,
    pub queued_samples: usize,
    pub seek: Option<Duration>,
    // 解码已正常结束，但队列和输出源仍可能持有未消费的样本。
    pub decode_finished: bool,
    pub error: Option<String>,
}

#[derive(Default)]
pub(super) struct Pcm {
    pub data: Mutex<PcmData>,
    pub changed: Condvar,
    pub control: Arc<StreamControl>,
    // 已交给输出回调的歌曲样本位置；缓冲静音不推进它。
    pub position_us: AtomicU64,
    // 输出端确认最后一个 PCM 已消费，控制器才允许自动切歌。
    pub output_drained: AtomicBool,
    // 解码已结束，排队的最后一块也已经交给输出端。
    pub closed_queue: AtomicBool,
    pub buffering: AtomicBool,
}

/// 单曲 PCM 消费端；缓冲欠载与正常曲终分别报告。
pub struct BufferedSource {
    pub(super) pcm: Arc<Pcm>,
    pub(crate) channels: u16,
    pub(crate) sample_rate: u32,
    pub(super) chunk: Option<Chunk>,
    pub(super) sample_index: usize,
    pub(crate) playback_gain: f32,
}

impl BufferedSource {
    fn pop_chunk(pcm: &Pcm, data: &mut PcmData) -> Option<Chunk> {
        let chunk = data.chunks.pop_front()?;
        data.queued_samples -= chunk.samples.len();
        if data.decode_finished {
            if data.chunks.is_empty() && data.error.is_none() {
                pcm.closed_queue.store(true, Ordering::Release);
            }
        } else {
            pcm.changed.notify_one();
        }
        Some(chunk)
    }

    /// 提前取出下一首的首个 PCM 块，交接时无需争抢解码器的锁。
    pub(crate) fn prime(&mut self) -> bool {
        let mut data = self.pcm.data.lock().unwrap();
        if data.error.is_some() || data.chunks.is_empty() {
            return false;
        }
        self.chunk = Self::pop_chunk(&self.pcm, &mut data);
        self.pcm.buffering.store(false, Ordering::Release);
        true
    }

    fn read_pcm_frame(&mut self, frame: &mut [f32]) -> FrameRead {
        assert_eq!(frame.len(), self.channels as usize);
        frame.fill(0.);
        if self.pcm.control.cancelled.load(Ordering::Acquire) {
            return FrameRead::Cancelled;
        }
        let generation = self.pcm.control.generation.load(Ordering::Acquire);
        if self
            .chunk
            .as_ref()
            .is_some_and(|chunk| chunk.generation != generation)
        {
            self.chunk = None;
        }
        let pcm = &self.pcm;
        if self.chunk.is_none()
            && let Ok(mut data) = pcm.data.try_lock()
        {
            let minimum =
                (self.sample_rate as f64 * self.channels as f64 * BUFFER_SECONDS) as usize;
            if self.pcm.buffering.load(Ordering::Acquire)
                && data.queued_samples < minimum
                && !data.decode_finished
                && data.error.is_none()
            {
                return FrameRead::Continue;
            }
            self.chunk = Self::pop_chunk(pcm, &mut data);
            self.sample_index = 0;
            let drained = self.chunk.is_none() && data.decode_finished && data.error.is_none();
            self.pcm.output_drained.store(drained, Ordering::Release);
            self.pcm
                .buffering
                .store(self.chunk.is_none() && !drained, Ordering::Release);
            if drained {
                return FrameRead::End;
            }
        } else if self.chunk.is_none() && pcm.closed_queue.load(Ordering::Acquire) {
            // 曲终已经确定；后台短暂持锁也无需在接缝插入静音。
            // 无锁读取不更新 output_drained，避免覆盖并发 seek 的重置。
            return FrameRead::End;
        }
        let Some(chunk) = &self.chunk else {
            return FrameRead::Continue;
        };
        frame.copy_from_slice(&chunk.samples[self.sample_index..self.sample_index + frame.len()]);
        self.sample_index += frame.len();
        if let Ok(_data) = self.pcm.data.try_lock()
            && self.pcm.control.is_current(chunk.generation)
        {
            let elapsed_us = self.sample_index as u64 / u64::from(self.channels) * 1_000_000
                / u64::from(self.sample_rate);
            self.pcm.position_us.store(
                chunk.position.as_micros() as u64 + elapsed_us,
                Ordering::Release,
            );
        }
        if self.sample_index == chunk.samples.len() {
            self.chunk = None;
        }
        FrameRead::Continue
    }

    #[cfg(test)]
    pub fn read_frame(&mut self, frame: &mut [f32]) -> bool {
        self.read_pcm_frame(frame) != FrameRead::Cancelled
    }

    /// 直接填充交错 f32 缓冲，并应用音量。
    #[cfg(test)]
    pub fn write(&mut self, output: &mut [f32], volume: f32) {
        for frame in output.chunks_mut(self.channels as usize) {
            self.read_frame(frame);
            for sample in frame {
                *sample = (*sample * volume).clamp(-1., 1.);
            }
        }
    }

    /// 渐变按声道帧推进，暂停淡出结束后停止消费 PCM。
    /// 正常曲终时返回本次回调的接续位置，其他情况返回缓冲区长度。
    pub(crate) fn write_playback(
        &mut self,
        output: &mut [f32],
        volume: f32,
        playing: bool,
    ) -> usize {
        let channels = self.channels as usize;
        let step = 1. / (self.sample_rate as f32 * FADE_SECONDS);
        for (index, frame) in output.chunks_mut(channels).enumerate() {
            if !playing && self.playback_gain == 0. {
                frame.fill(0.);
                continue;
            }
            if self.read_pcm_frame(frame) == FrameRead::End && playing {
                let end = index * channels;
                output[end..].fill(0.);
                return end;
            }
            self.playback_gain = if playing {
                (self.playback_gain + step).min(1.)
            } else {
                (self.playback_gain - step).max(0.)
            };
            let gain = volume * self.playback_gain;
            for sample in frame {
                *sample = (*sample * gain).clamp(-1., 1.);
            }
        }
        output.len()
    }
}

#[derive(PartialEq, Eq)]
enum FrameRead {
    Continue,
    End,
    Cancelled,
}
