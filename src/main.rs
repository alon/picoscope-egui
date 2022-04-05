use egui::*;
use eframe::egui;
use epi;
use plot::{
    Arrows, Bar, BarChart, BoxElem, BoxPlot, BoxSpread, CoordinatesFormatter, Corner, HLine,
    Legend, Line, LineStyle, MarkerShape, Plot, PlotImage, Points, Polygon, Text, VLine, Value,
    Values,
};

struct PicoPlot {

}


impl Widget for &mut PicoPlot {
    fn ui(self, ui: &mut Ui) -> Response {
        let plot = Plot::new("picoscope").legend(Legend::default());
        let markers = Points::new(Values::from_values(vec![Value::new(1.0, 1.0), Value::new(2.0, 2.0), Value::new(3.0, 0.0)])
            ).shape(MarkerShape::Circle)
            ;
        plot.show(ui, |plot_ui| {
            plot_ui.points(markers)
        }).response
    }
}

struct PicoScopeApp {
    plot: PicoPlot,
}

impl Default for PicoScopeApp {
    fn default() -> Self {
        PicoScopeApp { plot: PicoPlot {} }
    }
}

impl epi::App for PicoScopeApp {
    fn name(&self) -> &str {
        "Picoscope App"
    }

    fn update(&mut self, ctx: &egui::Context, frame: &epi::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Picoscope App");
            ui.add(&mut self.plot);
        });
        frame.set_window_size(ctx.used_size());
    }
}

fn main() {
    let app = PicoScopeApp::default();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(Box::new(app), native_options);
}
