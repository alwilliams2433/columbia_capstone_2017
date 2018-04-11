// Sets map

var map = L.map( 'map', {
    center: [39.171,-94.318],
    zoom: 4
});

L.tileLayer( 'https://api.mapbox.com/styles/v1/rcastellanes/cjftxvwef8m2o2rk3jirk1g1r/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1IjoicmNhc3RlbGxhbmVzIiwiYSI6IkVRd0dDaVEifQ.EsWXaa4BxE2qtGQ658rXqQ').addTo( map );

function getColor(d) {
    return d > 0.875 ? '#084594' :
        d > 0.750 ? '#2171b5' :
        d > 0.625 ? '#4292c6' :
        d > 0.500 ? '#6baed6' :
        d > 0.375 ? '#9ecae1' :
        d > 0.250 ? '#c6dbef' :
        d > 0.125 ? '#deebf7' :
        '#f7fbff';
}

function style(feature) {
    return {
        fillColor: getColor(features.properties.targ1pp),
        weight: 2,
        opacity: 1,
        color: 'white',
        dashArray: '3',
        fillOpacity: 0.7
    };
}

var geojson = L.geoJson(countydata, {style: style,}).addTo(map);

//var legend = L.control({position: 'bottomright'});
//               
//legend.onAdd = function (map) {
//    var div = L.DomUtil.create('div', 'info legend'),
//    grades = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
//    labels = [];
//    // loop through our density intervals and generate a label with a colored square for each interval
//    
//    for (var i = 0; i < grades.length; i++) {
//       div.innerHTML += '<i style="background:' + getColor(grades[i] +0.001) + '"></i> ' +
//       grades[i] + (grades[i + 1] ? '&ndash;' + grades[i + 1] + '<br>' : '+');
//    }
//       
//    return div;
//};
//    
//legend.addTo(map);

    }
}
