$(function() {
    var options = {
        parent: '#vis',
        chart: {
            'canvas': {
                'width': 800,
                'height': 800
            }
        }
    }
    
    var hexmap = new App.Models.HexMap();
    hexmap.set(options.chart);

    var view = new App.Views.InteractionLayer({
        el: options.parent,
        model: hexmap
    }).render();
});