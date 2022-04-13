use std::collections::HashMap;
use std::sync::Arc;
use egui::*;    
use eframe::egui;
use epi;
use plot::{
    Legend, Line, MarkerShape, Plot, Points, Value,
    Values,
};
use itertools::Itertools;

use pico_sdk::{
    download::{cache_resolution, download_drivers_to_cache},
    enumeration::{DeviceEnumerator, EnumResultHelpers},
};
use pico_sdk::prelude::*;

use crossbeam_channel::{unbounded, Sender, Receiver};
use pico_sdk::common::{
    PicoExtraOperations, PicoSigGenTrigSource, PicoSigGenTrigType, PicoSweepType, PicoWaveType, SweepShotCount
};

use rustfft::{FftPlanner, num_complex::Complex};
use pico_sdk::streaming::{SetSigGenArbitraryProperties, SetSigGenArbitraryPhaseProperties};


struct PicoScopeHandler {
    sender: Sender<StreamingEvent>,
}

impl NewDataHandler for PicoScopeHandler {
    fn handle_event(&self, event: &StreamingEvent) {
        // TODO: can we get rid of this clone?
        self.sender.send(event.clone()).unwrap();
    }
}

enum PlotDisplay {
    Time,
    FFT
}

#[derive(PartialEq)]
enum SigGenType {
    ShotPulse,
    SweepSine,
}

#[derive(Clone)]
enum SigGenEnum {
    ShotPulse { duration_usecs: f64, frequency_pulse: f64 }, // frequency_pulse of 10 and duration_usecs of 25 means 25us pulse every 100 = 1000 / 10.0 ms
    SweepSine { start_hz: f64, end_hz: f64, duration_secs: f64 }
}

impl Default for SigGenEnum {
    fn default() -> Self {
        SigGenEnum::ShotPulse {
            duration_usecs: 100.,
            frequency_pulse: 1000.,
        }
    }
}

#[derive(Clone)]
struct SigGen {
    pk_to_pk_microvolt: u32,
    offset_voltage_microvolt: i32,
    sig: SigGenEnum,
    trigger_type: PicoSigGenTrigType,
    trigger_source: PicoSigGenTrigSource,
}

impl Default for SigGen {
    fn default() -> Self {
        SigGen {
            pk_to_pk_microvolt: 3_000_000,
            offset_voltage_microvolt: 0,
            sig: Default::default(),
            trigger_type: PicoSigGenTrigType::Falling,
            trigger_source: PicoSigGenTrigSource::None,
        }
    }
}

#[derive(Debug, Clone)]
struct StreamProperties {
    sampling_rate: usize,
    a_range: PicoRange,
    b_range: PicoRange,
}

impl Default for StreamProperties {
    fn default() -> Self {
        StreamProperties {
            sampling_rate: 1_000_000,
            a_range: PicoRange::X1_PROBE_5V,
            b_range: PicoRange::X1_PROBE_5V,
        }
    }
}

trait AppScope {
    fn stop(
        &self,
    );

    fn set_siggen(
        &self,
        props: SetSigGenBuiltInV2Properties,
        handler: Arc<PicoScopeHandler>,
        stream_props: StreamProperties,
    ) -> PicoResult<()>;

    fn set_awg(
        &self,
        props: SetSigGenArbitraryProperties,
        handler: Arc<PicoScopeHandler>,
        stream_props: StreamProperties,
    ) -> PicoResult<()>;

    fn sig_gen_arbitrary_min_max_values(
        &self,
    ) -> PicoResult<SigGenArbitraryMinMaxValues>;

    fn trigger(&self);
}

struct AppPicoScope {
    stream_device: PicoStreamingDevice,
}

struct _AppSimulatedScope {}

impl AppScope for AppPicoScope {
    fn stop(
        &self,
    ) {
        self.stream_device.stop();
    }

    fn set_awg(
        &self,
        props: SetSigGenArbitraryProperties,
        handler: Arc<PicoScopeHandler>,
        stream_props: StreamProperties,
    ) -> PicoResult<()> {
        let ret = self.stream_device.set_sig_gen_arbitrary(props);

        self.stream_device.enable_channel(PicoChannel::A, stream_props.a_range, PicoCoupling::DC);
        self.stream_device.enable_channel(PicoChannel::B, stream_props.b_range, PicoCoupling::DC);
        // When handler goes out of scope, the subscription is dropped

        self.stream_device.new_data.subscribe(handler);
        self.stream_device.start(stream_props.sampling_rate as u32).unwrap();
        ret
    }

    fn sig_gen_arbitrary_min_max_values(&self) -> PicoResult<SigGenArbitraryMinMaxValues> {
        self.stream_device.sig_gen_arbitrary_min_max_values()
    }

    fn set_siggen(&self,
                  props: SetSigGenBuiltInV2Properties,
                  handler: Arc<PicoScopeHandler>,
                  stream_props: StreamProperties) -> PicoResult<()> {
        let ret = self.stream_device.set_sig_gen_built_in_v2(
            props.offset_voltage,
            props.pk_to_pk,
            props.wave_type,
            props.start_frequency,
            props.stop_frequency,
            props.increment,
            props.dwell_time,
            props.sweep_type,
            props.extra_operations,
            props.sweeps_shots,
            props.trig_type,
            props.trig_source,
            props.ext_in_threshold,
        );

        self.stream_device.enable_channel(PicoChannel::A, stream_props.a_range, PicoCoupling::DC);
        self.stream_device.enable_channel(PicoChannel::B, stream_props.b_range, PicoCoupling::DC);
        // When handler goes out of scope, the subscription is dropped

        self.stream_device.new_data.subscribe(handler);
        self.stream_device.start(stream_props.sampling_rate as u32).unwrap();
        ret
    }

    fn trigger(&self) {
        self.stream_device.sig_gen_software_control(1).unwrap();
        self.stream_device.sig_gen_software_control(0).unwrap();
    }
}

impl AppPicoScope {
    fn new() -> Self {
        let enumerator = DeviceEnumerator::with_resolution(cache_resolution());

        println!("Enumerating Pico devices...");
        let results = enumerator.enumerate();
        println!("pico enumerate: {:?}", results);

        let missing_drivers = results.missing_drivers();

        if !missing_drivers.is_empty() {
            println!(
                "Downloading drivers that failed to load {:?}",
                &missing_drivers
            );
            match download_drivers_to_cache(&missing_drivers) {
                Ok(_) => { },
                Err(_) => {
                    panic!("error downloading drivers");
                }
            }
            println!("Downloads complete");
        }

        let enum_device = results.into_iter().flatten().next().expect("No device found");
        let device = match enum_device.open() {
            Ok(device) => device,
            Err(err) => {
                panic!("error: {:?}", err);
            }
        };
        AppPicoScope { stream_device: device.into_streaming_device() }
    }
}

impl _AppSimulatedScope {
    // fn new() -> Self {
    //
    // }
}


struct SetSigGenBuiltInV2Properties {
    offset_voltage: i32, /* microvolts */
    pk_to_pk: u32,  /* microvolts */
    wave_type: PicoWaveType,
    start_frequency: f64, /* Hertz */
    stop_frequency: f64, /* Hertz */
    increment: f64, /* delta frequency jumps in Hertz */
    dwell_time: f64, /* amount to stay at each frequency in seconds */
    sweep_type: PicoSweepType,
    extra_operations: PicoExtraOperations,
    sweeps_shots: SweepShotCount,
    trig_type: PicoSigGenTrigType,
    trig_source: PicoSigGenTrigSource,
    ext_in_threshold: i16
}


impl PicoScopeApp {

    fn connect_to_scope(&mut self) {
        let simulate = std::env::var("PICOSCOPE_SIMULATE").unwrap_or("0".into()) == "1";
        if simulate {
            self.connect_to_simulation()
        } else {
            self.stream_device = Some(Box::new(AppPicoScope::new()));
        }
        self.reset_sig_gen_built_in().unwrap();
    }

    fn connect_to_simulation(&mut self) {
        /*
        let mut handler = self.handler.clone();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep_ms(10);
                let event = StreamingEvent {
                    length: 1,
                    samples_per_second: 1_000_000,
                    channels:
                };
                handler(event);
            }
        });
        */
        eprintln!("TODO");
    }

    fn reset_sig_gen_built_in(&mut self) -> PicoResult<()> {
        let sweeps_shots = SweepShotCount::None;

        let scope = self.stream_device.as_ref().unwrap();

        scope.stop();

        let res = match self.sig_gen.sig {
            SigGenEnum::ShotPulse { duration_usecs, frequency_pulse } => {
                // AWG of a pulse wave - short duration up, long duration down
                let dds_period_ns = 50.0;
                let total_duration_ns = 1e9 / frequency_pulse;
                // we want duration_usecs
                let min_max = scope.sig_gen_arbitrary_min_max_values()?;
                let max_size = min_max.max_size;
                let max_value = min_max.max_value;
                let single_sample_duration_ns = total_duration_ns / max_size as f64;
                let up_time_ns = duration_usecs * 1e3;
                println!("max awg allowed size: {} ; 1 sample period: {} = {} / {} (other: {:?}",
                    max_size,
                    single_sample_duration_ns, total_duration_ns, dds_period_ns,
                    min_max);
                let n = (up_time_ns / single_sample_duration_ns) as u32;
                println!("using {} samples, {} up, {} 0", max_size, n, max_size - n);
                let props = SetSigGenArbitraryProperties {
                    offset_voltage: 0,
                    pk_to_pk: self.sig_gen.pk_to_pk_microvolt,
                    phase_props: SetSigGenArbitraryPhaseProperties::FrequencyConstantHz(frequency_pulse),
                    arbitrary_waveform: (0..max_size).map(|x| (if x < n { max_value } else { 0 }) as i16).collect(),
                    sweep_type: PicoSweepType::Up,
                    extra_operations: PicoExtraOperations::Off,
                    sweeps_shots,
                    trig_type: PicoSigGenTrigType::Rising,
                    trig_source: PicoSigGenTrigSource::None,
                    ext_in_threshold: 0
                };
                scope.set_awg(props, self.handler.clone(), self.stream_props.clone())
            },
            SigGenEnum::SweepSine { start_hz, end_hz, duration_secs } => {
                // Built in signal generator
                let d_hz = end_hz - start_hz;
                let dwell = duration_secs / d_hz;
                let wave_type = PicoWaveType::Sine;
                let props = SetSigGenBuiltInV2Properties {
                    offset_voltage: self.sig_gen.offset_voltage_microvolt,
                    pk_to_pk: self.sig_gen.pk_to_pk_microvolt,
                    wave_type,
                    start_frequency: start_hz,
                    stop_frequency: end_hz,
                    increment: 1.0,
                    dwell_time: dwell,
                    sweep_type: PicoSweepType::Up,
                    extra_operations: PicoExtraOperations::Off,
                    sweeps_shots,
                    trig_type: self.sig_gen.trigger_type,
                    trig_source: self.sig_gen.trigger_source,
                    ext_in_threshold: 0,
                };
                scope.set_siggen(
                    props,
                    self.handler.clone(),
                    self.stream_props.clone())
            }
        };

        res
    }

    fn trigger(&self) {
        let stream_device = self.stream_device.as_ref().unwrap();
        stream_device.trigger();
    }
}

enum Trigger {
    None,
    Edge { val: i16, up: bool },
}

struct PicoScopeApp {
    num_points: u32,
    updates_per_second: f32,
    last_update_start: f64,
    last_update_count: usize,
    handler: Arc<PicoScopeHandler>,
    receiver: Receiver<StreamingEvent>,
    channels: HashMap<PicoChannel, Channel>,
    stream_device: Option<Box<dyn AppScope>>,
    stream_props: StreamProperties,

    // UI State
    show_t_or_fft: PlotDisplay,
    // how should we keep the state of the separate parameters for different modes?
    // mode a: x, y
    // mode b: z
    // when switching, I need to keep x, y and z around. keep each separately? a lot of code duplication
    start_hz: f64,
    end_hz: f64,
    sweep_duration_secs: f64,
    shot_duration_usec: f64,
    shot_frequency_hz: f64,
    sig_gen: SigGen,
    last_error: String,

    // TODO: bad name ; this is not the picoscope trigger, this is the level to use
    // for displaying the received wave ; we find the first instance where it matches
    // and use that as the zero of the axes.
    trigger: Trigger,
    trigger_level: i16,
}

struct Channel {
    name: String,
    multiplier: f64,
    samples: Vec<f64>
}

impl Default for PicoScopeApp {
    fn default() -> Self {
        let (sender, receiver) = unbounded();
        PicoScopeApp {
            num_points: 10000,
            last_update_count: 0,
            last_update_start: 0.0,
            updates_per_second: 0.0,
            receiver,
            handler: Arc::new(PicoScopeHandler { sender }),
            channels: HashMap::new(),
            stream_device: None,
            stream_props: Default::default(),
            show_t_or_fft: PlotDisplay::Time,
            sig_gen: Default::default(),
            last_error: "".into(),
            sweep_duration_secs: 1.0,
            shot_duration_usec: 25.0,
            shot_frequency_hz: 100.0,
            start_hz: 20_000.0,
            end_hz: 20_000.0,
            trigger: Trigger::Edge { val: 1000, up: true},
            trigger_level: 0,
        }
    }
}

impl PicoScopeApp {

}

impl epi::App for PicoScopeApp {
    fn name(&self) -> &str {
        "Picoscope App"
    }

    fn update(&mut self, ctx: &egui::Context, frame: &epi::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.ctx().request_repaint();
            // check for new data
            let mut reads = 0;
            loop {
                match self.receiver.try_recv() {
                    Err(_) => {
                        break;
                    },
                    Ok(data) => {
                        reads += 1;
                        for (pico_channel, c) in data.channels {
                            match self.channels.get_mut(&pico_channel) {
                                None => {
                                    let channel = Channel {
                                        name: format!("{}", pico_channel),
                                        multiplier: c.multiplier,
                                        samples: c.samples.iter().map(|x| *x as f64).collect()
                                    };
                                    self.channels.insert(pico_channel, channel);
                                },
                                Some(channel) => {
                                    channel.multiplier = c.multiplier;
                                    channel.name = format!("{}", pico_channel);
                                    // append points up to self.num_points, truncate first to drop old ones
                                    // so we keep to the length
                                    // not the best way
                                    let n = c.samples.len();
                                    let n_existing = channel.samples.len();
                                    if n > self.num_points as usize {
                                        // just take the new vector
                                        let start = n - self.num_points as usize;
                                        channel.samples = c.samples[start..].iter().map(|x| *x as f64).collect();
                                    } else if n + n_existing < self.num_points as usize {
                                        // pure concatenation
                                        channel.samples.extend(c.samples.iter().map(|x| *x as f64));
                                    } else {
                                        // A
                                        // =>
                                        // A[n-kept..] + B
                                        // total = num_points
                                        let kept = (self.num_points as usize) - n;
                                        channel.samples = channel.samples[(n_existing - kept)..]
                                            .iter().map(|x| *x).chain(c.samples.iter().map(|x| *x as f64)).collect();
        
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let min = self.channels.iter().map(|(_k, v)| v.samples.iter().cloned().fold(0./0., f64::min)).fold(0./0., f64::min);
            let max = self.channels.iter().map(|(_k, v)| v.samples.iter().cloned().fold(0./0., f64::max)).fold(0./0., f64::max);
            let now = ui.input().time;
            self.last_update_count += 1;
            let dt = now - self.last_update_start;
            if dt >= 1.0 {
                self.last_update_start = now;
                self.updates_per_second = self.last_update_count as f32 / dt as f32;
                self.last_update_count = 0;
            }
            //ui.heading("Picoscope App");
            let num_points = self.num_points;

            let mut show_t = match self.show_t_or_fft {
                PlotDisplay::Time => false,
                PlotDisplay::FFT => true,
            };

            // Top bar

            ui.horizontal(|ui| {
                ui.group(|ui| {
                    ui.add(egui::Label::new(format!("channels: {:2}; [{:06.0},{:06.0}]",
                        self.channels.len(), min, max)));
                    ui.add(egui::Label::new({
                        let temp: Vec<String> = self.channels
                            .iter()
                            .map(|(_pico_channel, c)| format!("{}-x{:01.9}", c.name, c.multiplier)).collect();
                        temp.join(" ")
                    }));

                    let mut sampling_rate = self.stream_props.sampling_rate;
                    egui::ComboBox::from_label("Sample")
                        .selected_text(format!("{:?}", sampling_rate))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut sampling_rate, 100_000, "0.1 Msps");
                            ui.selectable_value(&mut sampling_rate, 1_000_000, "1 Msps");
                            ui.selectable_value(&mut sampling_rate, 2_000_000, "2 Msps");
                        }
                    );

                    let mut trigger = match self.trigger {
                        Trigger::None => false,
                        _ => true,
                    };
                    ui.checkbox(&mut trigger, "Trigger");
                    if trigger {
                        ui.add(egui::Slider::new(&mut self.trigger_level, -32768..=32767).text("Level"));
                        self.trigger = Trigger::Edge { val: self.trigger_level, up: true };
                    } else {
                        self.trigger = Trigger::None;
                    }

                    let mut pk_to_pk_microvolt = self.sig_gen.pk_to_pk_microvolt;
                    egui::ComboBox::from_label("Sig Gen microvolts")
                        .selected_text(format!("{:?}", pk_to_pk_microvolt))
                        .show_ui(ui, |ui| {
                            for v in [1_000_000, 2_000_000, 3_000_000] {
                                ui.selectable_value(&mut pk_to_pk_microvolt, v, format!("{}", v));
                            };
                        });
                    self.sig_gen.pk_to_pk_microvolt = pk_to_pk_microvolt;
                    self.stream_props.sampling_rate = sampling_rate;
                    ui.checkbox(&mut show_t, "FFT");
                });

                self.show_t_or_fft = if show_t {
                    PlotDisplay::FFT
                } else {
                    PlotDisplay::Time
                };

                ui.group(|ui| {
                    ui.add(egui::Label::new(format!("fps: {:2.1}", self.updates_per_second)));
                    ui.add(egui::Slider::new(&mut self.num_points, 50..=100000).text("Points"));
                    ui.add(egui::DragValue::new(&mut self.num_points)
                        .speed((num_points as f64).max(2.0).log2())
                        .clamp_range(50..=100000)
                        .prefix("Points"));
                });
            });

            // error
            ui.label(&self.last_error);


            // Signal generator
            ui.horizontal(|ui| {
                if ui.button("Stop").clicked() {
                    self.stream_device.as_ref().unwrap().stop();
                }
                if ui.button("Update").clicked() {
                    self.last_error = match self.reset_sig_gen_built_in() {
                        Ok(_) => {
                            "".into()
                        },
                        Err(err) => {
                            format!("reset_sig_gen_built_in: {}: status = {}", err, err.status)
                        }
                    };
                }
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.sig_gen.trigger_source, PicoSigGenTrigSource::None, "No trigger");
                        ui.radio_value(&mut self.sig_gen.trigger_source, PicoSigGenTrigSource::None, "Software (BROKEN)"); // SoftTrig - but not working ; just use AWG for now
                    });
                });
                match self.sig_gen.trigger_source {
                    PicoSigGenTrigSource::SoftTrig => {
                        if ui.button("Trigger").clicked() {
                            self.trigger();
                        }
                    },
                    _ => {}
                }
                let mut sig_gen_type = match self.sig_gen.sig {
                    SigGenEnum::ShotPulse { .. } => SigGenType::ShotPulse,
                    SigGenEnum::SweepSine { .. } => SigGenType::SweepSine,
                };
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut sig_gen_type, SigGenType::ShotPulse, "Pulse square");
                        ui.radio_value(&mut sig_gen_type, SigGenType::SweepSine, "Sweep sine");
                    });
                });
                // Now recreate siggen.sig from sig_gen_type if they were switched
                self.sig_gen.sig = match &self.sig_gen.sig {
                    SigGenEnum::ShotPulse { .. } => match sig_gen_type {
                        SigGenType::SweepSine => SigGenEnum::SweepSine {
                            start_hz: self.start_hz, end_hz: self.end_hz, duration_secs: self.sweep_duration_secs },
                        _ => self.sig_gen.sig.clone(),
                    },
                    SigGenEnum::SweepSine { ..} => match sig_gen_type {
                        SigGenType::ShotPulse => SigGenEnum::ShotPulse {
                            duration_usecs: self.shot_duration_usec,
                            frequency_pulse: self.shot_frequency_hz,
                        },
                        _ => self.sig_gen.sig.clone(),
                    }
                };
                match &mut self.sig_gen.sig {
                    SigGenEnum::ShotPulse { ref mut duration_usecs, ref mut frequency_pulse } => {
                        ui.group(|ui| {
                            ui.add(egui::Label::new(format!("Duration [µs]: {}", self.shot_duration_usec)));
                            ui.add(egui::Slider::new(&mut self.shot_duration_usec, 1.0..=100.0).integer().text("µs"));
                            *duration_usecs = self.shot_duration_usec;
                            ui.add(egui::Label::new(format!("Frequency [Hz]: {:4.0}", self.shot_frequency_hz)));
                            ui.add(egui::Slider::new(&mut self.shot_frequency_hz, 10.0..=1000.0).integer().text("Hz"));
                            *frequency_pulse = self.shot_frequency_hz;
                        });
                    },
                    SigGenEnum::SweepSine { ref mut start_hz, ref mut end_hz, .. } => {
                        let mut start_hz_slider = *start_hz;
                        if *start_hz > *end_hz {
                            *end_hz = *start_hz;
                        }
                        let mut end_hz_slider = *end_hz;
                        ui.group(|ui| {
                            ui.add(egui::Label::new(format!("Start Freq: {:.0}", start_hz)));
                            ui.add(egui::Slider::new(&mut start_hz_slider, 10_000.0..=100_000.0).integer().text("Hz"));
                        });
                        ui.group(|ui| {
                            ui.add(egui::Label::new(format!("End Freq: {:.0}", end_hz)));
                            ui.add(egui::Slider::new(&mut end_hz_slider, start_hz_slider..=100_000.0).integer().text("Hz"));
                        });
                        // ui.group(|ui| {

                        // })
                        *start_hz = start_hz_slider;
                        *end_hz = end_hz_slider;
                    }
                };
            });

            // Main - Plot

            let plot = Plot::new("picoscope").legend(Legend::default());
                plot.show(ui, |plot_ui| {
                // TODO: For now just show the first channel
                if self.channels.len() == 0 {
                    // TODO: pass num_points to the simulation channel
                    // so it can be used to do stuff (amp / sin / cos)
                    let markers = Points::new(Values::from_values(
                        (0..(self.num_points + self.last_update_count as u32)).map(
                            |i| Value::new(i, ((i as f32) / 50.0).sin())).collect()
                    ))
                        .shape(MarkerShape::Circle)
                        ;
                    plot_ui.points(markers)
                } else {
                    let start_index = match self.trigger {
                        Trigger::None => 0,
                        Trigger::Edge {val, up} => {
                            self.channels.iter().map(|(_k, c)|
                                c.samples
                                    .iter()
                                    .tuple_windows()
                                    .enumerate()
                                    .skip_while(|(_i, (x_0, x_1))|
                                        // in wrong direction
                                        ((up && (**x_0 > **x_1)) && (!up && (**x_0 < **x_1)))
                                        // below the level
                                        || (**x_1 < (val as f64))
                                    )
                                    .map(|(i, (_x_0, _x_1))| i)
                                    .nth(0)
                            ).min().unwrap_or(Some(0)).unwrap_or(0)
                        }
                    };
                    for (_pico_channel, channel) in &self.channels {
                        match self.show_t_or_fft {
                            PlotDisplay::Time => {
                                let samples_it = channel.samples.iter().skip(start_index);
                                let values = Values::from_values(samples_it.enumerate().map(|(i, v)| Value {
                                    x: i as f64,
                                    y: *v,
                                }).collect());
                                //let markers = Points::new(values).shape(MarkerShape::Circle);
                                let line = Line::new(values);
                                plot_ui.line(line);
                            },
                            PlotDisplay::FFT => {
                                // plot fft - half the range, since it is real, so symmetric
                                let mut planner = FftPlanner::new();
                                let n = channel.samples.len();
                                let fft = planner.plan_fft_forward(n);
                                let mut buffer: Vec<_> = channel.samples.iter().map(|v| Complex::new(*v, 0.0)).collect();
                                fft.process(&mut buffer);
                                let dt = 1e-6;
                                let df = 1.0 / dt / (n as f64);
                                let values = Values::from_values(
                                    buffer.iter().take(n / 2).enumerate().map(|(i, v)| Value {
                                        x: i as f64 * df,
                                        y: (v.re * v.re + v.im * v.im).sqrt(),
                                    }).collect());
                                //let markers = Points::new(values).shape(MarkerShape::Circle);
                                let line = Line::new(values);
                                plot_ui.line(line);
                            }
                        }
                    }
                }
            }).response
        });
        frame.set_window_size(ctx.used_size());
    }
}

fn main() {
    if std::env::args().any(|a| a.contains("--trace")) {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ACTIVE)
            .init();
    }
    let mut app = PicoScopeApp::default();
    let native_options = eframe::NativeOptions::default();
    app.connect_to_scope();
    eframe::run_native(Box::new(app), native_options);
}
