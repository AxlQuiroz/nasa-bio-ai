console.log("app.js: Script cargado.");

// El código ahora se ejecuta directamente, ya que el script está al final del <body>

const form = document.getElementById('query-form');
const queryInput = document.getElementById('query');
const answerDiv = document.getElementById('answer');
const sourcesDiv = document.getElementById('sources');
const graphContainer = document.getElementById('knowledge-graph');
const thinkingSpan = document.getElementById('thinking');

// Verificación inicial para asegurarnos de que todo existe
if (!form || !queryInput || !answerDiv || !sourcesDiv || !graphContainer || !thinkingSpan) {
    console.error("CRÍTICO: Uno o más elementos del DOM no se encontraron. Verifica los IDs en tu index.html.");
} else {
    console.log("Todos los elementos del DOM fueron encontrados correctamente.");

    marked.setOptions({
        gfm: true,
        breaks: true,
        headerIds: false,
        mangle: false,
    });

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        thinkingSpan.style.display = 'inline';
        answerDiv.innerHTML = '';
        sourcesDiv.innerHTML = '';
        graphContainer.innerHTML = '';
        graphContainer.style.display = 'none';
        
        let fullResponse = "";
        let sources = [];

        try {
            // --- INICIO DE LA CORRECCIÓN ---

            // 1. Recoger las secciones seleccionadas (si existen)
            const selectedSections = [];
            document.querySelectorAll('input[name="section"]:checked').forEach((checkbox) => {
                selectedSections.push(checkbox.value);
            });

            // 2. Obtener el valor seleccionado del nuevo menú desplegable
            const analysisSelector = document.getElementById('analysis-type-selector');
            const selectedAnalysisType = analysisSelector.value;

            // 3. Construir el cuerpo de la petición incluyendo el tipo de análisis
            const requestBody = {
                query: query,
                sections: selectedSections,
                analysis_type: selectedAnalysisType // ¡Este es el nuevo campo!
            };

            // 4. Enviar la petición a la API correcta con la URL completa
            const response = await fetch('http://127.0.0.1:5000/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody) // Ahora requestBody sí existe
            });

            // --- FIN DE LA CORRECIÓN ---

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, {stream: true});
                const lines = chunk.split('\n').filter(line => line.trim() !== '');

                for (const line of lines) {
                    try {
                        // --- INICIO DE LA CORRECCIÓN ---
                        // Parsea cada línea directamente como un objeto JSON
                        const data = JSON.parse(line);
                        // --- FIN DE LA CORRECCIÓN ---

                        if (data.token === '[DONE]') {
                            thinkingSpan.style.display = 'none';
                            processFinalResponse(fullResponse, answerDiv);
                            displaySources(sources, sourcesDiv);
                            queryInput.value = '';
                            return;
                        }
                        if (data.sources) {
                            sources = data.sources;
                        } else if (data.graph) {
                            // Aquí puedes manejar la visualización del grafo
                            console.log("Datos del grafo recibidos:", data.graph);
                        } else if (data.token) {
                            fullResponse += data.token;
                            // Actualiza la respuesta en tiempo real si lo deseas
                            answerDiv.innerHTML = marked.parse(fullResponse);
                        }
                    } catch (e) {
                        console.error('Error parsing JSON from stream:', line, e);
                    }
                }
            }
        } catch (error) {
            console.error('Error fetching response:', error);
            thinkingSpan.style.display = 'none';
            answerDiv.innerHTML = '<p class="error">Error al obtener la respuesta del servidor.</p>';
        }
    });
}

function processFinalResponse(response, targetElement) {
    // --- AÑADE ESTA LÍNEA ---
    console.log("Respuesta completa recibida del backend:", response);
    // -------------------------

    const jsonStartIndex = response.indexOf('{');
    let textResponse = response;
    let graphData = null;

    if (jsonStartIndex !== -1) {
        textResponse = response.substring(0, jsonStartIndex).trim();
        const jsonString = response.substring(jsonStartIndex);
        try {
            const jsonData = JSON.parse(jsonString);
            if (jsonData.graph_data) {
                graphData = jsonData.graph_data;
            }
        } catch (e) {
            console.warn("Could not parse JSON from the end of the response.", e);
        }
    }

    targetElement.innerHTML = '';
    const responseP = document.createElement('p');
    responseP.innerHTML = marked.parse(textResponse);
    targetElement.appendChild(responseP);

    if (graphData && graphData.length > 0) {
        drawGraph(graphData);
    }
}

function drawGraph(graphData) {
    if (typeof vis === 'undefined') {
        console.error('vis.js library not found!');
        return;
    }

    const nodesSet = new Set();
    graphData.forEach(item => {
        nodesSet.add(item.source);
        nodesSet.add(item.target);
    });

    const nodes = new vis.DataSet(Array.from(nodesSet).map(node => ({
        id: node,
        label: node,
        shape: 'box',
        color: '#97C2FC',
        font: { size: 14 }
    })));

    const edges = new vis.DataSet(graphData.map(item => ({
        from: item.source,
        to: item.target,
        label: item.relationship,
        arrows: 'to',
        font: { align: 'middle', size: 12, color: '#555' },
        color: { color: '#848484' }
    })));

    const data = { nodes: nodes, edges: edges };

    const options = {
        layout: { improvedLayout: true },
        physics: {
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
                gravitationalConstant: -50,
                centralGravity: 0.01,
                springLength: 100,
                springConstant: 0.08,
                avoidOverlap: 0.5
            },
            stabilization: { iterations: 150 }
        },
        interaction: {
            dragNodes: true,
            dragView: true,
            zoomView: true
        }
    };

    graphContainer.style.display = 'block';
    new vis.Network(graphContainer, data, options);
}

function displaySources(sources, targetElement) {
    if (sources.length > 0) {
        const title = document.createElement('h3');
        title.textContent = 'Fuentes Consultadas:';
        targetElement.appendChild(title);

        const list = document.createElement('ul');
        sources.forEach(source => {
            const item = document.createElement('li');
            item.textContent = source;
            list.appendChild(item);
        });
        targetElement.appendChild(list);
    }
}

