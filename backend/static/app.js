document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('query-form');
  const queryInput = document.getElementById('query-input');
  const chatContainer = document.getElementById('chat-container');
  const thinkingIndicator = document.getElementById('thinking');
  const submitButton = document.getElementById('submit-button');

  let myChart = null; // Variable para guardar la instancia del gráfico

  // Opcional: ajustes de marked
  marked.setOptions({
    gfm: true,
    breaks: true,
    headerIds: false,
    mangle: false,
    highlight: (code, lang) => {
      try {
        return window.hljs ? hljs.highlight(code, { language: lang || 'plaintext' }).value : code;
      } catch { return code; }
    }
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    answerDiv.innerHTML = '';
    sourcesDiv.innerHTML = '';
    thinkingSpan.style.display = 'inline';

    let fullResponse = "";
    let sources = [];

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: query })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n').filter(line => line.trim() !== '');

            for (const line of lines) {
                if (line.startsWith('data:')) {
                    const dataStr = line.substring(5).trim();
                    try {
                        const data = JSON.parse(dataStr);

                        if (data.token === '[DONE]') {
                            thinkingSpan.style.display = 'none';
                            processFinalResponse(fullResponse, answerDiv);
                            displaySources(sources, sourcesDiv);
                            return;
                        }
                        if (data.sources) {
                            sources = data.sources;
                        } else if (data.token) {
                            fullResponse += data.token;
                        }
                    } catch (e) {
                        console.error('Error parsing JSON from stream:', dataStr, e);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error fetching response:', error);
        thinkingSpan.style.display = 'none';
        answerDiv.innerHTML = '<p class="error">Error al obtener la respuesta del servidor.</p>';
    }
});

function processFinalResponse(response, targetElement) {
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

    targetElement.innerHTML = ''; // Limpiar contenido anterior
    const responseP = document.createElement('p');
    responseP.innerHTML = marked.parse(textResponse);
    targetElement.appendChild(responseP);

    if (graphData) {
        console.log("Datos del Grafo de Conocimiento recibidos:", graphData);
        const graphTitle = document.createElement('h3');
        graphTitle.textContent = 'Conceptos Clave (Datos para el Grafo)';
        targetElement.appendChild(graphTitle);
        const graphPre = document.createElement('pre');
        graphPre.textContent = JSON.stringify(graphData, null, 2);
        targetElement.appendChild(graphPre);
    }
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

  submitButton.addEventListener('click', () => {
    // Limpia la respuesta anterior
    chatContainer.innerHTML = '';
    if (myChart) {
        myChart.destroy();
        document.getElementById('chart-container').classList.add('hidden');
    }
  });

  function addMessage(sender, markdownText) {
    const messageContainer = document.createElement('div');
    messageContainer.classList.add('message', `${sender}-message`);

    const textElement = document.createElement('p');
    // Sanitiza + convierte markdown a HTML seguro
    const rawHtml = marked.parse(markdownText || '');
    const safeHtml = (window.DOMPurify ? DOMPurify.sanitize(rawHtml) : rawHtml);
    textElement.innerHTML = safeHtml;

    messageContainer.appendChild(textElement);
    chatContainer.appendChild(messageContainer);
    smoothScroll();
    return messageContainer;
  }

  function appendMarkdown(targetP, chunkText) {
    // Simplemente añade el texto directamente para evitar que 'marked' cree nuevos párrafos por cada token.
    // Esto trata el texto como texto plano y lo añade al párrafo existente.
    const textNode = document.createTextNode(chunkText);
    targetP.appendChild(textNode);
  }

  function addSources(botMessageContainer, sources) {
    const sourcesContainer = document.createElement('div');
    sourcesContainer.classList.add('sources-container');

    const title = document.createElement('h4');
    title.textContent = 'Sources';
    sourcesContainer.appendChild(title);

    const list = document.createElement('ul');

    // Acepta string[] o objetos {title,url,source,year}
    sources.forEach((s, idx) => {
      const li = document.createElement('li');
      let title = '', href = '', meta = '';

      if (typeof s === 'string') {
        title = s;
      } else if (s && typeof s === 'object') {
        title = s.title || s.url || `Source #${idx+1}`;
        href = s.url || '';
        meta = [s.source, s.year].filter(Boolean).join(' • ');
      }

      if (href) {
        const a = document.createElement('a');
        a.href = href;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.textContent = title;
        li.appendChild(a);
      } else {
        li.textContent = title;
      }

      if (meta) {
        const small = document.createElement('small');
        small.style.marginLeft = '8px';
        small.textContent = meta;
        li.appendChild(small);
      }

      list.appendChild(li);
    });

    sourcesContainer.appendChild(list);
    botMessageContainer.appendChild(sourcesContainer);

  // Inserta un resumen de filtros aplicado (solo informativo)
  const fy = document.getElementById('filter-year')?.value || 'Todos los años';
  const ft = document.getElementById('filter-topic')?.value || 'Todos los temas';
  const fi = document.getElementById('filter-impact')?.value || 'Todos los impactos';
  const fp = document.getElementById('filter-progress-area')?.value || 'Todas las áreas de progreso';
  const fg = document.getElementById('filter-knowledge-gap')?.value || 'Todas las lagunas de conocimiento';
  const fc = document.getElementById('filter-consensus-area')?.value || 'Todas las áreas de consenso';
  const fd = document.getElementById('filter-disagreement-area')?.value || 'Todas las áreas de desacuerdo';

  addMessage('bot', `🔍 **Filtros aplicados**: ${fy} | ${ft} | ${fi} | ${fp} | ${fg} | ${fc} | ${fd}`);
  }

  function setThinking(show) {
    if (!thinkingIndicator) return;
    thinkingIndicator.style.display = show ? 'block' : 'none';
  }

  function smoothScroll() {
    chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
  }

  // util mínimo si no tienes DOMPurify (no reemplaza sanitización real)
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) =>
      ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c])
    );
  }
});

