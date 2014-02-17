$(function() {
    var options = {
        parent: '#vis',
        filePath: 'static/data/data.json'
    };

    d3.json(options.filePath, function(data) {
        var viewModel = new App.ViewModels.HexMapViewModel();
        var view = new App.Views.InteractionLayer({
            el: options.parent,
            model: viewModel,
            collection: data
        });
        view.render();
    });
});