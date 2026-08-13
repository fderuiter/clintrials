const CACHE_NAME = 'sim-hub-cache-v5';

const isSubpath = self.location.pathname.includes('/clintrials/');
const basePath = isSubpath ? '/clintrials/hub/' : '/hub/';

const urlsToCache = [
  basePath,
  basePath + 'index.html',
  basePath + 'manifest.json',
  basePath + 'icon.svg',
  basePath + 'vendor/iframeResizer.contentWindow.min.js',
  basePath + 'vendor/plotly-2.24.1.min.js',
  basePath + 'runner.py',
  basePath + 'schema.json',
  basePath + 'build-manifest.json'
];

self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(async cache => {
        // Cache the standard static assets first
        await cache.addAll(urlsToCache);

        // Dynamically fetch and cache the active wheel package from the manifest
        try {
          const manifestRes = await fetch(basePath + 'build-manifest.json');
          if (manifestRes.ok) {
            const manifestData = await manifestRes.json();
            if (manifestData && manifestData.wheel) {
              const dynamicWheel = basePath + manifestData.wheel;
              await cache.add(dynamicWheel);
            }
          }
        } catch (err) {
          console.error("Service worker failed to dynamically fetch/cache wheel from manifest:", err);
          // Fallback to cache the stable default wheel
          await cache.add(basePath + 'clintrials-0.1.4-py3-none-any.whl');
        }
      })
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName.startsWith('sim-hub-cache') && cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      return self.clients.claim();
    })
  );
});

self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  const path = url.pathname;

  const isCDN = url.hostname === 'cdn.jsdelivr.net' || url.hostname === 'cdn.plot.ly';
  if (!isCDN) {
    // Only intercept requests originating from within the Hub path
    if (!path.startsWith(basePath)) {
      return;
    }
  }

  // Restrict runtime caching and intercepting to GET requests only
  if (event.request.method !== 'GET') {
    return;
  }

  // Determine if this is a dynamic asset (schema.json, build-manifest.json, index.html, or base paths)
  const isDynamic = path.endsWith('/schema.json') ||
                    path.endsWith('/build-manifest.json') ||
                    path.endsWith('/index.html') ||
                    path === basePath ||
                    path === basePath + 'index.html' ||
                    path === basePath.slice(0, -1);

  if (isDynamic) {
    // Network-first strategy for dynamic assets
    event.respondWith(
      fetch(event.request)
        .then(response => {
          // Restrict runtime caching to successful responses only
          if (response && response.status === 200) {
            const responseToCache = response.clone();
            caches.open(CACHE_NAME).then(cache => {
              cache.put(event.request, responseToCache);
            });
          }
          return response;
        })
        .catch(() => {
          // Fallback to cache if network fails
          return caches.match(event.request).then(cachedResponse => {
            if (cachedResponse) {
              return cachedResponse;
            }
            // Propagate network error rather than returning undefined
            throw new Error('Network and cache failed for dynamic asset: ' + url.href);
          });
        })
    );
  } else {
    // Cache-first strategy for precached static assets
    event.respondWith(
      caches.match(event.request)
        .then(cachedResponse => {
          if (cachedResponse) {
            return cachedResponse;
          }

          return fetch(event.request).then(response => {
            // Restrict runtime caching to successful responses only
            if (response && response.status === 200) {
              const responseToCache = response.clone();
              caches.open(CACHE_NAME).then(cache => {
                cache.put(event.request, responseToCache);
              });
            }
            return response;
          }).catch(err => {
            // Fallback / revert to known stable local package if a wheel package is requested and fails to load offline
            if (url.pathname.endsWith('.whl')) {
              const fallbackWheel = basePath + 'clintrials-0.1.4-py3-none-any.whl';
              return caches.match(fallbackWheel).then(fallbackResponse => {
                if (fallbackResponse) {
                  return fallbackResponse;
                }
                return fetch(fallbackWheel);
              });
            }
            // Propagate network error rather than returning undefined
            throw err;
          });
        })
    );
  }
});
