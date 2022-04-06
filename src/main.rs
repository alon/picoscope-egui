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

struct PicoScopeHandler {
    sender: Sender<StreamingEvent>,
}

impl NewDataHandler for PicoScopeHandler {
    fn handle_event(&self, event: &StreamingEvent) {
        // TODO: can we get rid of this clone?
        self.sender.send(event.clone()).unwrap();
    }
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
    }

    fn connect_to_scope_real(&mut self) {
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
                    println!("error downloading drivers");
                    return;
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
                println!("error: {:?}", err);
                return;
            }
        };
        let stream_device = device.into_streaming_device();
        stream_device.enable_channel(PicoChannel::A, PicoRange::X1_PROBE_2V, PicoCoupling::DC);
        stream_device.enable_channel(PicoChannel::B, PicoRange::X1_PROBE_1V, PicoCoupling::AC);
        // When handler goes out of scope, the subscription is dropped

        stream_device.new_data.subscribe(self.handler.clone());
        stream_device.start(1_00_000).unwrap();
        self.stream_device = Some(stream_device);
    }

}

struct PicoScopeApp {
    num_points: u32,
    updates_per_second: f32,
    last_update_start: f64,
    last_update_count: usize,
    handler: Arc<PicoScopeHandler>,
    receiver: Receiver<StreamingEvent>,
    channels: Vec<Channel>,
    stream_device: Option<PicoStreamingDevice>,
    sampling_rate: usize,
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
            num_points: 100,
            last_update_count: 0,
            last_update_start: 0.0,
            updates_per_second: 0.0,
            receiver,
            handler: Arc::new(PicoScopeHandler { sender }),
            channels: vec![],
            stream_device: None,
            sampling_rate: 0,
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
                    self.channels = data.channels.iter().map(|(pico_channel, c)| {
                        Channel {
                            name: format!("{}", pico_channel),
                            multiplier: c.multiplier,
                            samples: c.samples.iter().map(|x| *x as f64).collect()
                        }
                    }).collect();
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
            ui.heading("Picoscope App");
            let num_points = self.num_points;
            ui.horizontal(|ui| {
                ui.group(|ui| {
                    ui.add(egui::Label::new(format!("channels: {}", self.channels.len())));
                    ui.add(egui::Label::new({
                        let temp: Vec<String> = self.channels
                            .iter()
                            .map(|c| format!("{}-x{}", c.name, c.multiplier)).collect();
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
                    if self.sampling_rate != sampling_rate {
                        println!("todo: change sampling rate to {}", sampling_rate)
                    }
                });
                ui.group(|ui| {
                    ui.add(egui::Label::new(format!("updates per second: {}", self.updates_per_second)));
                    ui.add(egui::Slider::new(&mut self.num_points, 50..=100000).text("Points"));
                    ui.add(egui::DragValue::new(&mut self.num_points)
                        .speed((num_points as f64).max(2.0).log2())
                        .clamp_range(50..=100000)
                        .prefix("Points"));
                });
            });
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
                    for channel in &self.channels {
                        let markers = Points::new(Values::from_values(
                            channel.samples.iter().enumerate().map(|(i, v)| Value {
                                x: i as f64,
                                y: *v,
                            }).collect()))
                                .shape(MarkerShape::Circle);
                        plot_ui.points(markers)    
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
