use std::collections::HashMap;
use std::sync::Arc;
use egui::*;    
use eframe::egui;
use epi;
use plot::{
    Arrows, Bar, BarChart, BoxElem, BoxPlot, BoxSpread, CoordinatesFormatter, Corner, HLine,
    Legend, Line, LineStyle, MarkerShape, Plot, PlotImage, Points, Polygon, Text, VLine, Value,
    Values,
};

use pico_sdk::{
    download::{cache_resolution, download_drivers_to_cache},
    enumeration::{DeviceEnumerator, EnumResultHelpers},
};
use pico_sdk::prelude::*;

use crossbeam_channel::{unbounded, Sender, Receiver};
use pico_sdk::common::{PicoExtraOperations, PicoIndexMode, PicoSigGenTrigSource, PicoSigGenTrigType, PicoSweepType, PicoWaveType, SweepShotCount};

use rustfft::{FftPlanner, num_complex::Complex};


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

impl PicoScopeApp {

    fn connect_to_scope(&mut self) {
        let simulate = std::env::var("PICOSCOPE_SIMULATE").unwrap_or("0".into()) == "1";
        if simulate {
            self.connect_to_simulation()
        } else {
            self.connect_to_scope_real()
        }
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
        panic!("TODO")
    }

    fn get_device(&mut self) -> PicoStreamingDevice {
        let enumerator = DeviceEnumerator::with_resolution(cache_resolution());

        println!("Enumerating Pico devices...");
        let results = enumerator.enumerate();

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
        let results = enumerator.enumerate();
        println!("pico enumerate: {:?}", results);

        let enum_device = results.into_iter().flatten().next().expect("No device found");
        let device = match enum_device.open() {
            Ok(device) => device,
            Err(err) => {
                panic!("error: {:?}", err);
            }
        };
        device.into_streaming_device()
    }

    fn connect_to_scope_real(&mut self) {
        self.stream_device = Some(self.get_device());
        self.reset_sig_gen_built_in().unwrap();
    }

    fn reset_sig_gen_built_in(&mut self) -> PicoResult<()> {
        let stream_device = self.stream_device.as_ref().unwrap();

        let sweeps_shots = match self.sig_gen.trigger_source {
            PicoSigGenTrigSource::None => SweepShotCount::None,
            _ => match self.sig_gen.sig {
                SigGenEnum::Shot { .. } => SweepShotCount::Shots(1),
                SigGenEnum::Sweep { .. } => SweepShotCount::ContinuousSweeps,
            }
        };

        let (start_freq, end_freq, dwell) = match self.sig_gen.sig {
            SigGenEnum::Shot { freq_hz, duration_secs } => (freq_hz, freq_hz, duration_secs),
            SigGenEnum::Sweep { start_hz, end_hz, duration_secs } => {
                let d_hz = end_hz - start_hz;
                (start_hz, end_hz, duration_secs / d_hz)
            }
        };

        // A shot with frequency 0 means a pulse of duration_secs time
        let wave_type = match self.sig_gen.sig {
            SigGenEnum::Sweep { .. } => {
                PicoWaveType::Sine
            },
            SigGenEnum::Shot { freq_hz, .. } => {
                if freq_hz == 0.0 {
                    PicoWaveType::DCVoltage
                } else {
                    PicoWaveType::Sine
                }
            }
        };

        // TODO: add arbitrary
        //stream_device.set_sig_gen_arbitrary();
        let res = stream_device.set_sig_gen_built_in_v2(
            self.sig_gen.offset_voltage_microvolt
            , self.sig_gen.pk_to_pk_microvolt
            , wave_type
            , start_freq
            , end_freq
            , 1.0
            , dwell
            , PicoSweepType::Up
            , PicoExtraOperations::Off
            , sweeps_shots
            , self.sig_gen.trigger_type
            , self.sig_gen.trigger_source
            , 0);


        stream_device.enable_channel(PicoChannel::A, PicoRange::X1_PROBE_2V, PicoCoupling::AC);
        stream_device.enable_channel(PicoChannel::B, PicoRange::X1_PROBE_1V, PicoCoupling::AC);
        // When handler goes out of scope, the subscription is dropped

        stream_device.new_data.subscribe(self.handler.clone());
        stream_device.start(self.sampling_rate as u32).unwrap();

        res
    }

    fn trigger(&self) {
        let stream_device = self.stream_device.as_ref().unwrap();
        stream_device.sig_gen_software_control(1);
        stream_device.sig_gen_software_control(0);
    }
}

#[derive(PartialEq)]
enum SigGenType {
    Shot,
    Sweep,
}

#[derive(Clone)]
enum SigGenEnum {
    Shot { freq_hz: f64, duration_secs: f64 },
    Sweep { start_hz: f64, end_hz: f64, duration_secs: f64 }
}

impl Default for SigGenEnum {
    fn default() -> Self {
        SigGenEnum::Sweep {
            start_hz: 20_000.0,
            end_hz: 40_000.0,
            duration_secs: 1.0,
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
            pk_to_pk_microvolt: 2_000_000,
            offset_voltage_microvolt: 0,
            sig: Default::default(),
            trigger_type: PicoSigGenTrigType::Falling,
            trigger_source: PicoSigGenTrigSource::None,
        }
    }
}

struct PicoScopeApp {
    num_points: u32,
    updates_per_second: f32,
    last_update_start: f64,
    last_update_count: usize,
    handler: Arc<PicoScopeHandler>,
    receiver: Receiver<StreamingEvent>,
    channels: HashMap<PicoChannel, Channel>,
    stream_device: Option<PicoStreamingDevice>,
    sampling_rate: usize,

    // UI State
    show_t_or_fft: PlotDisplay,
    sig_gen: SigGen,
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
            sampling_rate: 1_000_000,
            show_t_or_fft: PlotDisplay::Time,
            sig_gen: Default::default(),
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
            match self.receiver.try_recv() {
                Err(_) => {
                    // ignore
                },
                Ok(data) => {
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
                    ui.add(egui::Label::new(format!("channels: {}", self.channels.len())));
                    ui.add(egui::Label::new({
                        let temp: Vec<String> = self.channels
                            .iter()
                            .map(|(_pico_channel, c)| format!("{}-x{}", c.name, c.multiplier)).collect();
                        temp.join(" ")
                    }));
                    let mut sampling_rate = self.sampling_rate;
                    egui::ComboBox::from_label("Sampling rate")
                        .selected_text(format!("{:?}", sampling_rate))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut sampling_rate, 100_000, "0.1 Msps");
                            ui.selectable_value(&mut sampling_rate, 1_000_000, "1 Msps");
                            ui.selectable_value(&mut sampling_rate, 2_000_000, "2 Msps");
                        }
                    );
                    self.sampling_rate = sampling_rate;
                    ui.checkbox(&mut show_t, "FFT");
                });

                self.show_t_or_fft = if show_t {
                    PlotDisplay::FFT
                } else {
                    PlotDisplay::Time
                };

                ui.group(|ui| {
                    ui.add(egui::Label::new(format!("updates per second: {}", self.updates_per_second)));
                    ui.add(egui::Slider::new(&mut self.num_points, 50..=100000).text("Points"));
                    ui.add(egui::DragValue::new(&mut self.num_points)
                        .speed((num_points as f64).max(2.0).log2())
                        .clamp_range(50..=100000)
                        .prefix("Points"));
                });
            });

            
            // Signal generator
            ui.horizontal(|ui| {
                if ui.button("Update").clicked() {
                    match self.reset_sig_gen_built_in() {
                        Ok(_) => {},
                        Err(err) => {
                            ui.label(format!("{}", err));
                        }
                    }
                }
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.sig_gen.trigger_source, PicoSigGenTrigSource::None, "No trigger");
                        ui.radio_value(&mut self.sig_gen.trigger_source, PicoSigGenTrigSource::SoftTrig, "Software");
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
                    SigGenEnum::Shot { .. } => SigGenType::Shot,
                    SigGenEnum::Sweep { .. } => SigGenType::Sweep,
                };
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut sig_gen_type, SigGenType::Shot, "Shot");
                        ui.radio_value(&mut sig_gen_type, SigGenType::Sweep, "Sweep");
                    });
                });
                // Now recreate siggen.sig from sig_gen_type if they were switched
                let (start_freq, end_freq, duration) = 
                    match self.sig_gen.sig {
                        SigGenEnum::Shot { freq_hz, duration_secs } => (freq_hz, freq_hz, duration_secs),
                        SigGenEnum::Sweep { start_hz, end_hz, duration_secs } => (start_hz, end_hz, duration_secs),
                    };
                    
                self.sig_gen.sig = match (&self.sig_gen.sig, sig_gen_type) {
                    (SigGenEnum::Shot { .. }, SigGenType::Sweep) => {
                        SigGenEnum::Sweep { start_hz: start_freq, end_hz: end_freq, duration_secs: duration }
                    },
                    (SigGenEnum::Sweep { ..}, SigGenType::Shot) => {
                        SigGenEnum::Shot { freq_hz: start_freq, duration_secs: duration }
                    }
                    _ => { self.sig_gen.sig.clone() /* nothing to change */}
                };
                match &mut self.sig_gen.sig {
                    SigGenEnum::Shot { ref mut freq_hz, .. } => {
                        ui.group(|ui| {
                            let mut freq_slider = *freq_hz;
                            ui.add(egui::Label::new(format!("Freq: {}", freq_hz)));
                            ui.add(egui::Slider::new(&mut freq_slider, 10_000.0..=100_000.0).text("Hz"));
                            *freq_hz = freq_slider;
                        });
                    },
                    SigGenEnum::Sweep { ref mut start_hz, ref mut end_hz, .. } => {
                        let mut start_hz_slider = *start_hz;
                        let mut end_hz_slider = *end_hz;
                        ui.group(|ui| {
                            ui.add(egui::Label::new(format!("Start Freq: {}", start_hz)));
                            ui.add(egui::Slider::new(&mut start_hz_slider, 10_000.0..=100_000.0).text("Hz"));
                        });   
                        ui.group(|ui| {
                            ui.add(egui::Label::new(format!("End Freq: {}", end_hz)));
                            ui.add(egui::Slider::new(&mut end_hz_slider, start_hz_slider..=100_000.0).text("Hz"));
                        });
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
                    for (_pico_channel, channel) in &self.channels {
                        //plot_ui.points(markers)    
                        match self.show_t_or_fft {
                            PlotDisplay::Time => {
                                let values = Values::from_values(
                                    channel.samples.iter().enumerate().map(|(i, v)| Value {
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
    let mut app = PicoScopeApp::default();
    let native_options = eframe::NativeOptions::default();
    app.connect_to_scope();
    eframe::run_native(Box::new(app), native_options);
}
