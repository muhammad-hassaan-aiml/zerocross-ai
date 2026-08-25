/*! coi-serviceworker - Custom ZeroCross Auth Bypass v4 */
if (typeof window === 'undefined') {
    self.addEventListener("install", () => self.skipWaiting());
    self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

    self.addEventListener("fetch", function(e) {
        const req = e.request;
        if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
        
        // 1. Drop security completely for the Google Auth popup flow
        if (req.url.includes("auth=1")) {
            e.respondWith(fetch(req));
            return;
        }

        // 2. CRITICAL: Do not modify headers of external requests (Firebase, Tailwind)
        // Since our main document uses 'credentialless', it will allow these to load normally!
        if (!req.url.startsWith(self.location.origin)) {
            e.respondWith(fetch(req));
            return;
        }

        // 3. Apply COI headers ONLY to our local files (index.html, worker.js, etc)
        e.respondWith(fetch(req).then(res => {
            if (res.status === 0) return res; 
            
            const headers = new Headers(res.headers);
            headers.set("Cross-Origin-Embedder-Policy", "credentialless");
            headers.set("Cross-Origin-Resource-Policy", "cross-origin");
            headers.set("Cross-Origin-Opener-Policy", "same-origin"); 
            
            return new Response(res.body, {
                status: res.status,
                statusText: res.statusText,
                headers: headers
            });
        }).catch(e => console.error("COI ServiceWorker Fetch Error:", e)));
    });
} else {
    const isAuthMode = window.location.search.includes("auth=1");
    // If we are NOT in Auth Mode, we require strict multithreading security
    if (!isAuthMode && !window.crossOriginIsolated && window.isSecureContext) {
        if (navigator.serviceWorker) {
            navigator.serviceWorker.register(window.document.currentScript.src).then(reg => {
                reg.addEventListener("updatefound", () => window.location.reload());
                if (reg.active && !navigator.serviceWorker.controller) window.location.reload();
            });
        }
    }
}