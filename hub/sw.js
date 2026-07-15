const CACHE_NAME = 'sim-hub-cache-v3';

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
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Cache hit - return response
        if (response) {
          return response;
        }

        // IMPORTANT: Clone the request. A request is a stream and
        // can only be consumed once.
        let fetchRequest = event.request.clone();

        return fetch(fetchRequest).then(
          response => {
            // Check if we received a valid response
            if(!response || response.status !== 200 || response.type !== 'basic') {
              // We could also cache CORS responses (type 'cors' or 'opaque') from the CDN
              if (response && response.status === 200) {
                 let responseToCache = response.clone();
                 caches.open(CACHE_NAME)
                   .then(cache => {
                     cache.put(event.request, responseToCache);
                   });
              }
              return response;
            }

            let responseToCache = response.clone();

            caches.open(CACHE_NAME)
              .then(cache => {
                cache.put(event.request, responseToCache);
              });

            return response;
          }
        ).catch(err => {
            console.log('Fetch failed, offline?', err);
        });
      })
  );
});
