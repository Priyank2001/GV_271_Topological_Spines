// The comparison box of compare mode (side panel): what differs between the two
// shown extraction variants, and why.
//
// It only renders; main.js gathers the numbers (compare.js) and calls ``render``.
// The "why" part lists the design choices (``traits``, from index.json) on which
// the two variants differ, so a new variant explains itself by declaring its
// traits in comparison/variants.py.

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const int = (n) => Number(n).toLocaleString('en-US');
const pct = (x) => `${(100 * x).toFixed(x > 0.999 || x < 0.001 ? 0 : 1)} %`;

/** One line of facts about a variant's full graph, from its ``notes`` (index.json). */
export function noteText(info) {
  const n = info?.notes ?? {};
  const parts = [];
  if (n.maxima !== undefined) {
    parts.push(`${int(n.maxima)} maxima and ${int(n.saddles)} saddles` +
               (info.simplifiable ? ' before simplification' : ', all shown'));
  }
  if (n.loop_saddles) parts.push(`${int(n.loop_saddles)} saddles close a loop`);
  if (n.saddles_reaching_3plus) parts.push(`${int(n.saddles_reaching_3plus)} saddles reach 3 or more maxima`);
  if (n.pieces > 1) parts.push(`the graph falls into ${int(n.pieces)} separate pieces`);
  if (n.raw_maxima !== undefined) {
    parts.push(`on the stored values (which have ties) it would report ${int(n.raw_maxima)} maxima` +
               (n.preprocess_maxima !== undefined ? `, ${int(n.preprocess_maxima)} after its own preprocessing script` : '') +
               '; shown here on values made distinct');
  }
  return parts.join('; ');
}

export class ComparisonBox {
  constructor(element) { this.el = element; }

  hide() { this.el.hidden = true; }

  /**
   * @param {object} p
   * @param {object} p.infoA, p.infoB  variant records from index.json (label, traits, notes, simplifiable)
   * @param {object} p.shown           result of compareShown() for what is drawn now
   * @param {object|null} p.agreement  findAgreement(): the full graphs compared at build time
   * @param {Array} p.diffs            traitDifferences()
   * @param {string} [p.selection]     a sentence about the current selection, if any
   */
  render({ infoA, infoB, shown, agreement, diffs, selection = '' }) {
    const A = esc(infoA?.label ?? 'left'), B = esc(infoB?.label ?? 'right');
    const html = [];
    html.push(`<div class="title">Comparing <span class="who">${A}</span> (left) and ` +
              `<span class="who">${B}</span> (right)</div>`);

    html.push(`<p><b>Shown now:</b> ${int(shown.shownA)} peaks left, ${int(shown.shownB)} right; ` +
              `${int(shown.shared)} at the same vertex in both. ` +
              `<span class="mark-key"></span> ${int(shown.missingA.length)} left and ${int(shown.missingB.length)} ` +
              'right are not maxima at all in the other method.</p>');

    if (agreement) {
      html.push(`<p><b>Full graphs</b> (before any simplification): maxima Jaccard ${agreement.maxima_jaccard.toFixed(3)} ` +
                `(${int(agreement.maxima_only_a)} only left, ${int(agreement.maxima_only_b)} only right); ` +
                `${pct(agreement.region_agreement)} of the points climb to the same maximum; ` +
                `linked pairs of maxima: Jaccard ${agreement.pair_jaccard.toFixed(3)} ` +
                `(${int(agreement.pairs_only_a)} only left, ${int(agreement.pairs_only_b)} only right).</p>`);
    }
    if (selection) html.push(`<p class="sel">${selection}</p>`);

    html.push('<div class="why"><b>Why they differ</b>');
    if (diffs.length) {
      html.push('<dl>' + diffs.map((d) =>
        `<dt>${esc(d.label)}</dt><dd><span class="who">${A}:</span> ${esc(d.a)}<br>` +
        `<span class="who">${B}:</span> ${esc(d.b)}</dd>`).join('') + '</dl>');
    } else {
      html.push('<p>The two methods make the same design choices.</p>');
    }
    html.push('</div>');

    const notes = [[A, noteText(infoA)], [B, noteText(infoB)]].filter(([, t]) => t);
    if (notes.length) {
      html.push('<div class="facts"><b>Full graphs</b><ul>' +
                notes.map(([who, t]) => `<li><span class="who">${who}:</span> ${esc(t)}</li>`).join('') + '</ul></div>');
    }
    this.el.innerHTML = html.join('');
    this.el.hidden = false;
  }
}
