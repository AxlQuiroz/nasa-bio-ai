document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const chatContainer = document.getElementById('chat-container');
    const thinkingIndicator = document.getElementById('thinking');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        addMessage('user', query);
        queryInput.value = '';
        thinkingIndicator.style.display = 'block';

        const botMessageContainer = addMessage('bot', '');
        const botTextElement = botMessageContainer.querySelector('p');

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: query }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = '';
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                thinkingIndicator.style.display = 'none';

                if (done) {
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                
                for (let i = 0; i < lines.length - 1; i++) {
                    const line = lines[i];
                    if (line.startsWith('data:')) {
                        const jsonString = line.substring(5).trim();
                        try {
                            const data = JSON.parse(jsonString);
                            if (data.token) {
                                if (data.token === '[DONE]') {
                                    return;
                                }
                                fullResponse += data.token;
                                botTextElement.innerHTML = marked.parse(fullResponse);
                            } else if (data.sources) {
                                addSources(botMessageContainer, data.sources);
                            }
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        } catch (e) {
                            console.error('Error parsing JSON:', jsonString);
                        }
                    }
                }
                buffer = lines[lines.length - 1];
            }

        } catch (error) {
            console.error('Error:', error);
            botTextElement.innerHTML = 'Error: Could not connect to the server for a streaming response.';
            thinkingIndicator.style.display = 'none';
        }
    });

    function addMessage(sender, text) {
        const messageContainer = document.createElement('div');
        messageContainer.classList.add('message', `${sender}-message`);
        
        const textElement = document.createElement('p');
        // Renderiza el texto inicial usando marked
        textElement.innerHTML = marked.parse(text); 
        
        messageContainer.appendChild(textElement);
        chatContainer.appendChild(messageContainer);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return messageContainer;
    }

    function addSources(botMessageContainer, sources) {
        const sourcesContainer = document.createElement('div');
        sourcesContainer.classList.add('sources-container');
        
        const title = document.createElement('h4');
        title.textContent = 'Sources:';
        sourcesContainer.appendChild(title);

        const list = document.createElement('ul');
        sources.forEach(source => {
            const item = document.createElement('li');
            item.textContent = source;
            list.appendChild(item);
        });
        sourcesContainer.appendChild(list);
        botMessageContainer.appendChild(sourcesContainer);

    // Filtros dinámicos
    const year = document.getElementById('filter-year').value;
    const topic = document.getElementById('filter-topic').value;
    const impact = document.getElementById('filter-impact').value;

    const filterSummary = `🔍 Filtros aplicados: 
        ${year || "Todos los años"} | 
        ${topic || "Todos los temas"} | 
        ${impact || "Todos los impactos"}`;
    
    addMessage('bot', filterSummary);
    }
});
