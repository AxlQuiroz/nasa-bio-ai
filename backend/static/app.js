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

    // Lee filtros (sácalos del DOM aquí)
    const year = (document.getElementById('filter-year')?.value || '').trim();
    const topic = (document.getElementById('filter-topic')?.value || '').trim();
    const impact = (document.getElementById('filter-impact')?.value || '').trim();
    const progressArea = (document.getElementById('filter-progress-area')?.value || '').trim();
    const knowledgeGap = (document.getElementById('filter-knowledge-gap')?.value || '').trim();
    const consensusArea = (document.getElementById('filter-consensus-area')?.value || '').trim();
    const disagreementArea = (document.getElementById('filter-disagreement-area')?.value || '').trim();

    addMessage('user', escapeHtml(query));
    queryInput.value = '';
    setThinking(true);

    const botMessageContainer = addMessage('bot', '');
    const botTextElement = botMessageContainer.querySelector('p');

    try {
      const response = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: query,
          filters: {
            year,
            topic,
            impact,
            progressArea,
            knowledgeGap,
            consensusArea,
            disagreementArea
          } // <<<<<< manda filtros al backend
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }
      if (!response.body) {
        throw new Error('No streaming body (response.body === null)');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      // Parser SSE robusto
      let buffer = '';
      let gotFirstToken = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Soporta \r\n y eventos SSE multi-línea:
        let eventBoundary;
        while ((eventBoundary = buffer.indexOf('\n\n')) !== -1) {
          const rawEvent = buffer.slice(0, eventBoundary);
          buffer = buffer.slice(eventBoundary + 2);

          const lines = rawEvent.split(/\r?\n/);
          let eventType = 'message';
          let dataLines = [];

          for (const line of lines) {
            if (!line) continue;
            if (line.startsWith(':')) continue; // comentario/keep-alive
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataLines.push(line.slice(5).trim());
            }
          }

          const dataStr = dataLines.join('\n');
          if (!dataStr) continue;

          if (!gotFirstToken) {
            // Oculta “pensando” en el PRIMER token válido
            setThinking(false);
            gotFirstToken = true;
          }

          // Cada evento debe ser un JSON
          let data;
          try {
            data = JSON.parse(dataStr);
          } catch (err) {
            console.warn('JSON chunk inválido:', dataStr);
            continue;
          }

          // Protocolo: { token?: string, sources?: [...], meta?: {...} }
          if (data.token) {
            if (data.token === '[DONE]') {
              setThinking(false);
              break; // salimos del while interno; el externo romperá al terminar el stream
            }
            // --- LÍNEA CLAVE AÑADIDA ---
            // Reemplaza cualquier salto de línea (\n, \r) por un espacio
            const cleanToken = data.token.replace(/(\r\n|\n|\r)/gm, " ");
            
            // Usamos el token limpio para añadirlo al HTML
            appendMarkdown(botTextElement, cleanToken);
            smoothScroll();
          }

          if (Array.isArray(data.sources) && data.sources.length) {
            // Renderiza fuentes una sola vez por mensaje
            if (!botMessageContainer.querySelector('.sources-container')) {
              addSources(botMessageContainer, data.sources);
            }
          }
        }
      }

      setThinking(false);

      // Opcional: si no hubo tokens, muestra aviso
      if (!botTextElement.innerText.trim()) {
        botTextElement.textContent = 'No recibí contenido. Intenta ajustar tu consulta o filtros.';
      }

    } catch (error) {
      console.error('Stream error:', error);
      botTextElement.textContent = 'Error: no pude conectar con el servidor de streaming.';
      setThinking(false);
    }
  });

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

