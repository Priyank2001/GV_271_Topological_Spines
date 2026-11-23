// The second spine of compare mode: another extraction variant of the same
// field, drawn next to the main panel.
//
// It owns its SpineView, its document and what is visible, and knows how to
// select "the same node" as in the other panel (same vertex). Everything else
// (sliders, info panel, comparison box) stays in main.js, which treats this
// panel as one more view to keep in step.

import { SpineView } from './spineView.js';
import { findVisible, visibleGraph } from './compare.js';

export class ComparePanel {
  /**
   * @param {HTMLElement} container element the spine canvas goes into
   * @param {{onHover?: Function, onSelect?: Function, onCamera?: Function}} handlers: the first two
   *        get this panel's node, onCamera the view's camera after a pan, zoom or reset
   * @param {Function} [makeView] builds the view (tests pass a fake); default: a WebGL SpineView
   */
  constructor(container, handlers = {}, makeView = null) {
    this.container = container;
    this.handlers = handlers;
    this.makeView = makeView ?? ((el, h) => new SpineView(el, h));
    this.view = null;           // created on first use: no WebGL context until compare mode opens
    this.doc = null;
    this.vis = null;
    this.quiet = false;         // true while selecting on behalf of the other panel
  }

  /** The SpineView, created on first call; ``onCreate`` gets it to copy the drawing settings. */
  ensureView(onCreate) {
    if (!this.view) {
      this.view = this.makeView(this.container, {
        onHover: (node, pos) => this.handlers.onHover?.(node, pos),
        onSelect: (node) => { if (!this.quiet) this.handlers.onSelect?.(node); },
        onCamera: (camera) => this.handlers.onCamera?.(camera),
      });
      onCreate?.(this.view);
    }
    return this.view;
  }

  setDocument(doc) {
    this.doc = doc;
    this.vis = null;
    this.view.setGraph(doc);
  }

  /** Apply the sliders (ignored by methods without simplification, see ``visibleGraph``). */
  setThresholds(threshold, variation) {
    if (!this.doc) return;
    this.vis = visibleGraph(this.doc, threshold, variation);
    this.view.setVisible(this.vis);
  }

  /** Vertex of a node of this panel (``{kind, docId}``), or null. */
  vertexOf(node) {
    if (!node || !this.doc) return null;
    return (node.kind === 'peak' ? this.doc.peaks : this.doc.saddles)[node.docId].vertex;
  }

  /**
   * Select the visible node of ``kind`` at ``vertex`` without echoing the
   * selection back. Clears the selection when there is none; returns whether
   * one was found.
   */
  selectVertex(kind, vertex) {
    if (!this.vis) return false;
    const index = vertex === null ? -1 : findVisible(this.doc, this.vis, kind, vertex);
    this.quiet = true;
    try {
      this.view.select(index < 0 ? null : { kind, index });
    } finally {
      this.quiet = false;
    }
    return index >= 0;
  }
}
