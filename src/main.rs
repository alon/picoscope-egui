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
        println!("Sample count: {}", event.length);
        // TODO: can we get rid of this clone?
        self.sender.send(event.clone()).unwrap();
    }
}



impl PicoScopeApp {

    fn connect_to_scope(&mut self) -> Result<Vec<std::result::Result<pico_sdk::prelude::EnumeratedDevice, pico_sdk::prelude::EnumerationError>>, Box<dyn std::error::Error>> {
        let enumerator = DeviceEnumerator::with_resolution(cache_resolution());

        println!("Enumerating Pico devices...");
        let results = enumerator.enumerate();

        let missing_drivers = results.missing_drivers();

        if !missing_drivers.is_empty() {
            println!(
                "Downloading drivers that failed to load {:?}",
                &missing_drivers
            );
            download_drivers_to_cache(&missing_drivers)?;
            println!("Downloads complete");
            let enum_device = results.into_iter().flatten().next().expect("No device found");
            let device = enum_device.open()?;
            let stream_device = device.into_streaming_device();
            stream_device.enable_channel(PicoChannel::A, PicoRange::X1_PROBE_2V, PicoCoupling::DC);
            stream_device.enable_channel(PicoChannel::B, PicoRange::X1_PROBE_1V, PicoCoupling::AC);
            // When handler goes out of scope, the subscription is dropped

            stream_device.new_data.subscribe(self.handler.clone());
            stream_device.start(1_000)?;
        }
        let results = enumerator.enumerate();
        Ok(results)
    }

}

struct PicoScopeApp {
    num_points: u32,
    updates_per_second: f32,
    last_update_start: f64,
    last_update_count: usize,
    handler: Arc<PicoScopeHandler>,
    receiver: Receiver<StreamingEvent>
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
            match self.receiver.recv() {
                Err(_) => {
                    // ignore
                },
                Ok(data) => {
                    println!("todo: handle streaming event from scope: {}", data.length);
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
            ui.group(|ui| {
                ui.add(egui::Label::new(format!("updates per second: {}", self.updates_per_second)));
                ui.add(egui::Slider::new(&mut self.num_points, 50..=100000).text("Points"));
                ui.add(egui::DragValue::new(&mut self.num_points)
                    .speed((num_points as f64).max(2.0).log2())
                    .clamp_range(50..=100000)
                    .prefix("Points"));
            });
            let plot = Plot::new("picoscope").legend(Legend::default());
            let markers = Points::new(Values::from_values(
                (0..(self.num_points + self.last_update_count as u32)).map(|i| Value::new(i, ((i as f32) / 50.0).sin())).collect()
            ))
                .shape(MarkerShape::Circle)
                ;
            plot.show(ui, |plot_ui| {
                plot_ui.points(markers)
            }).response
        });
        frame.set_window_size(ctx.used_size());
    }
}

fn main() {
    let mut app = PicoScopeApp::default();
    let native_options = eframe::NativeOptions::default();
    println!("pico enumerate: {:?}", app.connect_to_scope());
    eframe::run_native(Box::new(app), native_options);
}
