/* docs/_static/custom.js */

// 0. Global Setup: Load iframe resizer immediately and prevent ReferenceErrors on load
if (typeof window.iFrameResize === 'undefined') {
    const stubIframeResize = function (...args) {
        const retry = () => {
            if (window.iFrameResize && window.iFrameResize !== stubIframeResize) {
                window.iFrameResize(...args);
            } else if (window.iframeResize && window.iframeResize !== stubIframeResize) {
                window.iframeResize(...args);
            } else {
                setTimeout(retry, 50);
            }
        };
        setTimeout(retry, 50);
    };
    window.iFrameResize = window.iframeResize = stubIframeResize;
}
(function loadIframeResizer() {
    if (document.getElementById('iframe-resizer-script')) return;
    const script = document.createElement('script');
    script.id = 'iframe-resizer-script';
    
    function getIframeResizerUrl() {
        const path = window.location.pathname;
        if (path.includes('/clintrials/')) {
            return window.location.origin + '/clintrials/_static/vendor/iframeResizer.min.js';
        } else {
            return '/_static/vendor/iframeResizer.min.js';
        }
    }
    
    script.src = getIframeResizerUrl();
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

    const mobileCloseBtn = document.createElement('button');
    mobileCloseBtn.id = 'hub-mobile-close-btn';
    mobileCloseBtn.innerHTML = '✕ Close Simulation';
    mobileCloseBtn.setAttribute('aria-label', 'Close Simulation Hub');

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

    sidebar.appendChild(mobileCloseBtn);
    sidebar.appendChild(resizer);
    sidebar.appendChild(iframeContainer);

    document.body.appendChild(sidebar);
    document.body.appendChild(toggleBtn);

    // Initialize any existing iframes on the page
    window.iFrameResize({
        log: false,
        checkOrigin: false,
        heightCalculationMethod: 'lowestElement'
    }, 'iframe');

    // 2. State & Functions
    let isOpen = false;
    let savedScrollY = 0;

    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

    function getQueryParam(name) {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get(name);
    }

    function deserializeState(str) {
        if (!str) return null;
        try {
            return JSON.parse(decodeURIComponent(str));
        } catch (e) {
            try {
                return JSON.parse(atob(str));
            } catch (e2) {
                return null;
            }
        }
    }

    function setUrlState(state) {
        const url = new URL(window.location.href);
        if (state && state.model) {
            const stateStr = encodeURIComponent(JSON.stringify(state));
            url.searchParams.set('sim_state', stateStr);
        } else {
            url.searchParams.delete('sim_state');
        }
        
        const urlStr = url.toString();
        if (window.location.href !== urlStr) {
            history.pushState({ sim_state: state }, '', urlStr);
        }
    }

    const debouncedSetUrlState = debounce(setUrlState, 300);

    function getHubUrl() {
        const path = window.location.pathname;
        if (path.includes('/clintrials/')) {
            return window.location.origin + '/clintrials/hub/';
        } else {
            return '/hub/';
        }
    }

    function initIframeResizer() {
        if (window.innerWidth < 768) return;
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
        iframe.title = 'Clinical Trials Simulation Hub Dashboard';
        
        const stateStr = getQueryParam('sim_state');
        if (stateStr) {
            iframe.src = getHubUrl() + '?sim_state=' + stateStr;
        } else {
            iframe.src = getHubUrl();
        }
        iframeContainer.appendChild(iframe);
        initIframeResizer();
    }

    function openSidebar() {
        if (window.innerWidth < 768) {
            savedScrollY = window.scrollY;
            document.body.classList.add('hub-drawer-open');
        }
        isOpen = true;
        sidebar.classList.add('open');
        createIframe();
    }

    function closeSidebar(syncUrl = true) {
        isOpen = false;
        sidebar.classList.remove('open');
        if (window.innerWidth < 768) {
            document.body.classList.remove('hub-drawer-open');
            window.scrollTo(0, savedScrollY);
        }
        if (syncUrl) {
            setUrlState(null);
        }
    }

    // 3. Toggle Logic
    toggleBtn.addEventListener('click', (e) => {
        e.preventDefault();
        if (isOpen) {
            closeSidebar(true);
        } else {
            openSidebar();
        }
    });

    mobileCloseBtn.addEventListener('click', (e) => {
        e.preventDefault();
        closeSidebar(true);
    });

    // 4. Message & Popstate listeners
    window.addEventListener('message', (event) => {
        if (event.origin !== window.location.origin) return;
        if (event.data && event.data.type === 'simulationState') {
            debouncedSetUrlState(event.data.state);
        }
    });

    window.addEventListener('popstate', (event) => {
        const stateStr = getQueryParam('sim_state');
        if (stateStr) {
            const state = deserializeState(stateStr);
            if (state) {
                if (!isOpen) {
                    openSidebar();
                }
                const iframe = document.getElementById('simulation-hub-iframe');
                if (iframe && iframe.contentWindow) {
                    iframe.contentWindow.postMessage({ type: 'restoreState', state }, window.location.origin);
                }
            }
        } else {
            if (isOpen) {
                closeSidebar(false);
            }
        }
    });

    // Handle initial state on load
    const initialStateStr = getQueryParam('sim_state');
    if (initialStateStr) {
        openSidebar();
    }

    // 5. Resize Logic
    let isResizing = false;
    let startX, startWidth;

    resizer.addEventListener('mousedown', (e) => {
        if (window.innerWidth <= 768) return;
        e.preventDefault();
        isResizing = true;
        startX = e.clientX;
        startWidth = sidebar.offsetWidth;
        document.body.classList.add('hub-dragging');
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        const dx = startX - e.clientX;
        let newWidth = startWidth + dx;
        
        const minWidth = 350;
        const maxWidth = window.innerWidth * 0.8;
        
        if (newWidth < minWidth) newWidth = minWidth;
        if (newWidth > maxWidth) newWidth = maxWidth;
        
        sidebar.style.width = `${newWidth}px`;
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.classList.remove('hub-dragging');
        }
    });

    // Homepage Placeholder & Click-to-Play
    const homepagePlaceholder = document.getElementById('homepage-sim-placeholder');
    if (homepagePlaceholder) {
        // Disable side drawer button on homepage
        toggleBtn.style.display = 'none';

        const launchBtn = document.getElementById('launch-sim-btn');
        const loadingSpinner = document.getElementById('sim-loading-spinner');

        if (launchBtn) {
            launchBtn.addEventListener('click', () => {
                launchBtn.style.display = 'none';
                if (loadingSpinner) loadingSpinner.style.display = 'flex';

                const iframe = document.createElement('iframe');
                iframe.title = 'Clinical Trials Simulation Hub Dashboard';
                iframe.src = getHubUrl() + 'index.html?embed=true&view=Win+Ratio';
                iframe.style.width = '100%';
                iframe.style.height = '800px';
                iframe.style.border = 'none';
                iframe.style.background = 'transparent';
                iframe.style.overflow = 'hidden';
                iframe.setAttribute('scrolling', 'no');
                iframe.style.opacity = '0';
                iframe.style.transition = 'opacity 0.3s ease-in-out';

                iframe.onload = () => {
                    if (loadingSpinner) loadingSpinner.style.display = 'none';
                    iframe.style.opacity = '1';
                    
                    Array.from(homepagePlaceholder.children).forEach(child => {
                        if (child !== iframe) {
                            child.style.display = 'none';
                        }
                    });
                    
                    homepagePlaceholder.style.border = 'none';
                    homepagePlaceholder.style.background = 'transparent';
                    homepagePlaceholder.style.boxShadow = 'none';
                    
                    if (window.iFrameResize) {
                        window.iFrameResize({
                            log: false,
                            checkOrigin: false,
                            heightCalculationMethod: 'lowestElement'
                        }, iframe);
                    }
                };

                homepagePlaceholder.appendChild(iframe);
            });
        }
    }

    // 5. Service Worker Unregistration & Cache Cleanup
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.getRegistrations().then(registrations => {
            for (const registration of registrations) {
                const isHubScope = registration.scope.endsWith('/hub/') || registration.scope.includes('/hub/');
                if (!isHubScope) {
                    registration.unregister().then(success => {
                        if (success) {
                            console.log('Successfully unregistered stale root-scoped service worker:', registration.scope);
                        }
                    });
                }
            }
        }).catch(err => {
            console.error('Error getting service worker registrations:', err);
        });

        if ('caches' in window) {
            caches.delete('sim-hub-cache-v4').then(deleted => {
                if (deleted) {
                    console.log('Successfully cleared stale service worker cache (sim-hub-cache-v4)');
                }
            });
        }
    }
});
