const CACHE_NAME = 'sim-hub-cache-v4';

const isSubpath = self.location.pathname.includes('/clintrials/');
const basePath = isSubpath ? '/clintrials/hub/' : '/hub/';

const urlsToCache = [
  basePath,
  basePath + 'index.html',
  basePath + 'manifest.json',
  basePath + 'icon.svg',
  basePath + 'vendor/stlite.css',
  basePath + 'vendor/iframeResizer.contentWindow.min.js',
  basePath + 'vendor/stlite.js'
];

self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        return cache.addAll(urlsToCache);
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
  // Restrict runtime caching and intercepting to GET requests only
  if (event.request.method !== 'GET') {
    return;
  }

  const url = new URL(event.request.url);
  const path = url.pathname;

  // Determine if this is a dynamic asset (schema.json, index.html, or base paths)
  const isDynamic = path.endsWith('/schema.json') ||
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
            // Propagate network error rather than returning undefined
            throw err;
          });
        })
    );
  }
});
