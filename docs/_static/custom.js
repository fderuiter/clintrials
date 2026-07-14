/* docs/_static/custom.js */

// 0. Global Setup: Load iframe resizer immediately and prevent ReferenceErrors on load
if (typeof window.iFrameResize === 'undefined') {
    window.iFrameResize = function stubIframeResize(...args) {
        const retry = () => {
            if (window.iFrameResize && window.iFrameResize !== stubIframeResize) {
                window.iFrameResize(...args);
            } else {
                setTimeout(retry, 50);
            }
        };
        setTimeout(retry, 50);
    };
}
(function loadIframeResizer() {
    if (document.getElementById('iframe-resizer-script')) return;
    const script = document.createElement('script');
    script.id = 'iframe-resizer-script';
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.min.js';
    script.async = true;
    document.head.appendChild(script);
})();

document.addEventListener('DOMContentLoaded', () => {
    // 1. Inject DOM Elements
    const toggleBtn = document.createElement('button');
    toggleBtn.id = 'hub-toggle-btn';
    toggleBtn.innerHTML = '🧪';
    toggleBtn.title = 'Toggle Simulation Hub';
    toggleBtn.setAttribute('aria-label', 'Toggle Simulation Hub');

    const sidebar = document.createElement('div');
    sidebar.id = 'hub-sidebar';

    const resizer = document.createElement('div');
    resizer.id = 'hub-resizer';
    resizer.innerHTML = `
        <div class="hub-resizer-grip">
            <div class="hub-resizer-line"></div>
            <div class="hub-resizer-line"></div>
        </div>
    `;

    const iframeContainer = document.createElement('div');
    iframeContainer.id = 'hub-iframe-container';

    sidebar.appendChild(resizer);
    sidebar.appendChild(iframeContainer);

    document.body.appendChild(sidebar);
    document.body.appendChild(toggleBtn);

    // 2. State & Functions
    let isOpen = false;

    function getHubUrl() {
        const path = window.location.pathname;
        if (path.includes('/clintrials/')) {
            return window.location.origin + '/clintrials/hub/';
        } else {
            return '/hub/';
        }
    }

    function initIframeResizer() {
        if (window.iFrameResize) {
            window.iFrameResize({
                log: false,
                checkOrigin: false,
                heightCalculationMethod: 'lowestElement'
            }, '#simulation-hub-iframe');
        } else {
            setTimeout(initIframeResizer, 100);
        }
    }

    function createIframe() {
        if (document.getElementById('simulation-hub-iframe')) return;

        const iframe = document.createElement('iframe');
        iframe.id = 'simulation-hub-iframe';
        iframe.src = getHubUrl();
        iframeContainer.appendChild(iframe);
        initIframeResizer();
    }

    function destroyIframe() {
        const iframe = document.getElementById('simulation-hub-iframe');
        if (iframe) {
            if (iframe.iFrameResizer) {
                iframe.iFrameResizer.close();
            } else {
                iframe.parentNode.removeChild(iframe);
            }
        }
    }

    // 3. Toggle Logic
    toggleBtn.addEventListener('click', (e) => {
        e.preventDefault();
        isOpen = !isOpen;

        if (isOpen) {
            sidebar.classList.add('open');
            createIframe();
        } else {
            sidebar.classList.remove('open');
            destroyIframe();
        }
    });

    // 4. Resize Logic
    let isResizing = false;
    let startX, startWidth;

    resizer.addEventListener('mousedown', (e) => {
        e.preventDefault();
        isResizing = true;
        startX = e.clientX;
        startWidth = sidebar.offsetWidth;
        document.body.classList.add('hub-dragging');
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        const dx = startX - e.clientX;
        const newWidth = startWidth + dx;
        sidebar.style.width = `${newWidth}px`;
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.classList.remove('hub-dragging');
        }
    });
});
