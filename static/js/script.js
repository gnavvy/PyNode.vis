$(function() {
    var viewModel = new App.ViewModels.HexMapViewModel();
    var dataSeries = new App.Collections.DataSeries();
    new App.Views.HexMapView({
        el: '#vis',
        model: viewModel,
        collection: dataSeries
    }).render();

    setInterval(function() {
        dataSeries.fetch({
            reset: true
        });
    }, 5000);
});