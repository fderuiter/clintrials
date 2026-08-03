// docs/_extra/sw.js - Kill-switch service worker to force-unregister and clear deadlocked cache.

const CACHE_TO_CLEAR = 'sim-hub-cache-v4';

self.addEventListener('install', event => {
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.delete(CACHE_TO_CLEAR)
      .then(() => self.registration.unregister())
      .then(() => self.clients.matchAll({ type: 'window' }))
      .then(clients => {
        clients.forEach(client => {
          if (client.navigate) {
            client.navigate(client.url);
          }
        });
      })
  );
});
