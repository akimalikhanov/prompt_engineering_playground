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

