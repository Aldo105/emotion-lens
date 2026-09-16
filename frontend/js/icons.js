/**
 * EmotionLens — iconos SVG
 *
 * Sustituyen a los emojis, que se veían como una aplicación escolar y además
 * se dibujan distinto en cada sistema operativo: el mismo emoji cambia de
 * forma y de color entre Windows, Android y iOS, así que la interfaz nunca se
 * veía igual dos veces.
 *
 * Todos heredan el color del texto mediante currentColor, de modo que se
 * adaptan solos al tema claro y al oscuro.
 *
 * Uso:  Icons.render('check')  ->  cadena con el <svg>
 *       Icons.dot('#8098BE')   ->  punto de color para una emoción
 */
const Icons = (() => {
    const svg = (contenido, opciones = {}) => {
        const t = opciones.size || 18;
        const trazo = opciones.stroke || 2;
        return `<svg class="icon" width="${t}" height="${t}" viewBox="0 0 24 24" `
            + `fill="none" stroke="currentColor" stroke-width="${trazo}" `
            + `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">`
            + `${contenido}</svg>`;
    };

    const trazos = {
        // Estado
        check:      '<circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.5 2.5 4.5-5"/>',
        alert:      '<circle cx="12" cy="12" r="9"/><path d="M12 7.5v5"/><circle cx="12" cy="16" r=".6" fill="currentColor"/>',
        cross:      '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>',
        clock:      '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
        dotCircle:  '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3.5" fill="currentColor" stroke="none"/>',

        // Navegación y vistas
        activity:   '<path d="M3 12h4l3 8 4-16 3 8h4"/>',
        sessions:   '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M8 4v5"/>',
        compare:    '<path d="M9 4v16M15 4v16"/><path d="M4 8l3-3 3 3M20 16l-3 3-3-3"/>',
        upload:     '<path d="M12 16V4M8 8l4-4 4 4"/><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3"/>',
        video:      '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3z"/>',
        settings:   '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/>',

        // Métricas
        heart:      '<path d="M12 20s-7-4.5-7-9.5A3.8 3.8 0 0112 8a3.8 3.8 0 017 2.5c0 5-7 9.5-7 9.5z"/>',
        pulse:      '<path d="M3 12h3l2-5 4 10 2-5h7"/>',
        face:       '<circle cx="12" cy="12" r="9"/><circle cx="9" cy="10" r=".8" fill="currentColor" stroke="none"/><circle cx="15" cy="10" r=".8" fill="currentColor" stroke="none"/><path d="M8.5 14.5c1 1.2 2.1 1.8 3.5 1.8s2.5-.6 3.5-1.8"/>',
        spark:      '<path d="M12 3l1.8 5.4L19 10l-5.2 1.6L12 17l-1.8-5.4L5 10l5.2-1.6z"/>',
        target:     '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/>',
        chart:      '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',

        // Acciones
        play:       '<path d="M7 4l12 8-12 8z"/>',
        stop:       '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
        note:       '<path d="M5 3h10l4 4v14H5z"/><path d="M15 3v4h4M9 12h6M9 16h4"/>',
        download:   '<path d="M12 4v11M8 11l4 4 4-4"/><path d="M4 19h16"/>',
        document:   '<path d="M6 2h8l4 4v16H6z"/><path d="M14 2v4h4M9 13h6M9 17h6"/>',
        trash:      '<path d="M4 7h16M10 7V5h4v2M6 7l1 14h10l1-14"/>',
        refresh:    '<path d="M4 12a8 8 0 0113.6-5.7L20 9"/><path d="M20 4v5h-5"/><path d="M20 12a8 8 0 01-13.6 5.7L4 15"/><path d="M4 20v-5h5"/>',
        eye:        '<path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z"/><circle cx="12" cy="12" r="2.5"/>',
        moon:       '<path d="M20 14.5A8.5 8.5 0 019.5 4a8.5 8.5 0 1010.5 10.5z"/>',
        sun:        '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
        folder:     '<path d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2z"/>',
        microscope: '<path d="M7 18h10M9 18V9a3 3 0 016 0v9"/><path d="M12 4v2M5 21h14"/>',
    };

    return {
        render(nombre, opciones) {
            const d = trazos[nombre];
            return d ? svg(d, opciones) : '';
        },
        /** Punto de color: para emociones, más sobrio que una carita. */
        dot(color, tam = 10) {
            return `<span class="emo-dot" style="--dot:${color};width:${tam}px;height:${tam}px"></span>`;
        },
        /** Inserta iconos en todo [data-icon] del documento. */
        apply(raiz = document) {
            raiz.querySelectorAll('[data-icon]').forEach((el) => {
                const nombre = el.getAttribute('data-icon');
                const html = this.render(nombre, { size: el.dataset.iconSize || 18 });
                if (html) el.innerHTML = html;
            });
        },
    };
})();

if (typeof window !== 'undefined') {
    window.Icons = Icons;
    document.addEventListener('DOMContentLoaded', () => Icons.apply());
}
