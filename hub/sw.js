const CACHE_NAME = 'sim-hub-cache-v1';
const urlsToCache = [
  './',
  './index.html',
  './manifest.json',
  './icon.svg',
  'https://cdn.jsdelivr.net/npm/@stlite/mountable@0.55.0/build/stlite.css',
  'https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js',
  'https://cdn.jsdelivr.net/npm/@stlite/mountable@0.55.0/build/stlite.js'
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
