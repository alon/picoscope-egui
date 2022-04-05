use egui::*;
use eframe::egui;
use epi;
use plot::{
    Arrows, Bar, BarChart, BoxElem, BoxPlot, BoxSpread, CoordinatesFormatter, Corner, HLine,
    Legend, Line, LineStyle, MarkerShape, Plot, PlotImage, Points, Polygon, Text, VLine, Value,
    Values,
};

struct PicoScopeApp {
    num_points_text: u32,
    updates_per_second: f32,
    last_update_start: std::time::Instant,
    last_update_count: usize,
}

impl Default for PicoScopeApp {
    fn default() -> Self {
        PicoScopeApp {
            num_points_text: 100,
            last_update_count: 0,
            last_update_start: std::time::Instant::now(),
            updates_per_second: 0.0,
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
            self.last_update_count += 1;
            let now = std::time::Instant::now();
            let dt = now - self.last_update_start;
            if dt >= std::time::Duration::from_secs(1) {
                self.last_update_start = now;
                self.updates_per_second = self.last_update_count as f32 / dt.as_secs_f32();
                self.last_update_count = 0;
            }
            ui.heading("Picoscope App");
            ui.add(egui::Label::new(format!("updates per second: {}", self.updates_per_second)));
            ui.add(egui::Slider::new(&mut self.num_points_text, 50..=10000).text("Points"));
            let plot = Plot::new("picoscope").legend(Legend::default());
            let markers = Points::new(Values::from_values(
                (0..(self.num_points_text + self.last_update_count as u32)).map(|i| Value::new(i, ((i as f32) / 50.0).sin())).collect()
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
    let app = PicoScopeApp::default();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(Box::new(app), native_options);
}
