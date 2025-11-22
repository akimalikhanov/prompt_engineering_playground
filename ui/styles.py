"""CSS styles for the Gradio UI."""

BADGES_CSS = """
.badges {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    align-items: center;
    margin-top: 0.5rem;
}
.badge {
    display: inline-flex;
    align-items: baseline;
    gap: 0.35rem;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    border: 1px solid rgba(15, 23, 42, 0.16);
}
.badge-label {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.7rem;
}
.badge-value {
    font-weight: 600;
}
@media (prefers-color-scheme: dark) {
    .badge {
        border-color: rgba(148, 163, 184, 0.35);
    }
    .badge-label {
    }
    .badge-value {
    }
}
"""

TEXTAREA_SCROLL_CSS = """
/* Make all Textbox textareas scrollable with a reasonable max height */
.scroll-text textarea {
    max-height: 320px;      /* adjust as needed */
    overflow: auto !important;
    resize: vertical;       /* allow user to expand if they want */
}
"""

ENHANCED_UI_CSS = """
/* Enhanced UI checkbox styling - attention-grabbing */
.enhanced-ui-checkbox {
    font-size: 16px !important;
    font-weight: 600 !important;
}

.enhanced-ui-checkbox label {
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Chatbot message action buttons: ensure Copy -> Like -> Dislike order */
button[aria-label*="Copy"] {
    order: 1;
}

button[aria-label*="Like"] {
    order: 2;
}

button[aria-label*="Dislike"] {
    order: 3;
}

/* Add explicit spacing between buttons (works regardless of parent layout) */
button[aria-label*="Like"] {
    margin-left: 0.5rem !important;
}

button[aria-label*="Dislike"] {
    margin-left: 0.5rem !important;
}
"""

NOTE_CSS = """
/* Note styling - smaller and slightly transparent */
.note-style {
    font-size: 0.9rem !important;
    opacity: 0.9;
    color: rgba(0, 0, 0, 0.8);
}

@media (prefers-color-scheme: dark) {
    .note-style {
        color: rgba(255, 255, 255, 0.8);
    }
}

.note-style p {
    margin: 0.25rem 0;
    font-size: 0.9rem !important;
    opacity: 0.9;
}

.note-style strong {
    opacity: 0.9;
    font-weight: 500;
}
"""

