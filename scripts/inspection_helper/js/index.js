const color = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffa9ff', '#000000'];
const createCharts = (contexts, data) => {
    let result = {};
    for (let k in contexts) {
        if (!data.hasOwnProperty(k)) {
            console.warn(`A context for '{k}' provided without any data.`);
            continue;
        }
        const addExtraDatasetSpecs = (d, i) => ({...d,
            hidden: true,
            backgroundColor: "transparent",
            borderColor: color[i % color.length],
            pointBackgroundColor: color[i % color.length],
            pointBorderColor: color[i % color.length],
            pointHoverBackgroundColor:color[i % color.length],
            pointHoverBorderColor: color[i % color.length],
        });
        result[k] = new Chart(contexts[k], {
            type: 'line',
            data: { datasets: data[k].map(addExtraDatasetSpecs) },
            options: {
                scales: {
                    x: {
                        type: 'linear'
                    }
                }
            }
        });
        // result[k] = new Chart(contexts[k], {
        //     type: 'line',
        //     data: { datasets: data[k] }
        // });
        console.log({
            type: 'line',
            data: { datasets: data[k] }
        });
    }
    return result;
};
document.addEventListener('DOMContentLoaded', () => {
    let contexts = {
        'loss': document.getElementById('loss_chart').getContext('2d'),
        'validation_loss': document.getElementById('validation_loss_chart').getContext('2d'),
        'validation_bleu': document.getElementById('validation_bleu_chart').getContext('2d'),
    };
    let charts = {};
    fetch('./data.json').then(r => r.json()).then(r => {
        charts = createCharts(contexts, r);
    });
});