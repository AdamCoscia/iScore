export default class perturbation {
  constructor(utils, elem) {
    this.utils = utils;
    this.div = elem;
    this.node = elem.node();
    this.tooltip = d3.select("#charts-tooltip");
    this.layers = {
      interactions: null,
      drawing: null,
    };
    this.data = null;
    this.controls = {
      summarySelectedID: "1",
      render: false,
      model: "",
      span: "",
      colorBy: "",
    };
  }

  init() {
    const self = this;

    const div = self.div;

    // add groups in layer order (i.e., draw element groups in this order)
    const interactionsLayer = div.append("div").attr("class", "interaction-layer");
    const drawingLayer = interactionsLayer.append("div").attr("class", "drawing-layer");

    // save groups to access later
    self.layers.interactions = interactionsLayer;
    self.layers.drawing = drawingLayer;
  }

  clear() {
    this.layers.drawing.selectAll("*").remove();
  }

  render() {
    const self = this;
    const show = document.getElementById("show-perturbation-checkbox").checked;

    if (!self.controls.render || !show) return;

    const fw = parseFloat(self.div.style("width"));
    const fh = parseFloat(self.div.style("height"));

    const selectedSummaryData = self.data[self.controls.summarySelectedID];

    let scores, data;
    if (self.controls.span == "word") {
      scores = selectedSummaryData.perturbationScores[self.controls.model][self.controls.span].out;
      data = selectedSummaryData.perturbationScores[self.controls.model][self.controls.span].nested;
    } else {
      scores = selectedSummaryData.perturbationScores[self.controls.model][self.controls.span];
      data = selectedSummaryData.perturbationScores[self.controls.model][self.controls.span];
    }
    data.forEach((x, i) => (x["idx"] = i));

    // Create color scale
    let colorScale;
    const trueValueExtent = d3.extent(scores.map((x) => x.diff_true));
    const absValueExtent = d3.extent(scores.map((x) => x.diff_abs));
    if (self.controls.colorBy == "diff_true") {
      const lb = trueValueExtent[0];
      const ub = trueValueExtent[1];
      let extent;
      if (lb < 0 && ub > 0) {
        extent = [trueValueExtent[0], 0, trueValueExtent[1]];
        colorScale = d3.scaleDiverging(extent, (t) => d3.interpolateRdBu(t));
      } else if (lb >= 0 && ub >= 0) {
        extent = trueValueExtent;
        colorScale = d3.scaleSequential(extent, (t) => d3.interpolateBlues(t));
      } else if (lb <= 0 && ub <= 0) {
        extent = trueValueExtent;
        colorScale = d3.scaleSequential(extent, (t) => d3.interpolateReds(1 - t));
      }
    } else if (self.controls.colorBy == "diff_sum_norm" || self.controls.colorBy == "diff_max_norm") {
      colorScale = d3.scaleSequential([0, 1], d3.interpolatePurples);
    } else {
      colorScale = d3.scaleSequential(absValueExtent, d3.interpolatePurples);
    }

    // draw token/word/sentence spans
    const spanDivs = self.layers.drawing
      .selectAll("div")
      .data(data)
      .join("div")
      .attr("data-index", (x) => {
        if (!x.hasOwnProperty("_children") || x._children.length == 0) return `${x["idx"]}`;
      })
      .attr("class", (x) => {
        const isBreak = x.hasOwnProperty("type") && x.type == "break";
        const hasSynonyms = self.controls.span == "word" && x._children.length > 0;
        return isBreak ? "break" : hasSynonyms ? "synonyms-wrapper" : self.controls.span;
      })
      .style("color", "black")
      .html((x) => {
        let html = "";
        const span = x[self.controls.span];
        switch (self.controls.span) {
          case "word":
            html = x._children.length == 0 ? cleanSpan(span) : "";
            break;
          case "grammar":
          case "sentence":
          case "token":
            const isBreak = x.hasOwnProperty("type") && x.type == "break";
            html = isBreak ? span : cleanSpan(span);
            break;
        }
        return html;
      });

    if (self.controls.span == "word") {
      const synonymsWrappers = spanDivs.filter((x) => x._children.length > 0);
      synonymsWrappers
        .append("span")
        .attr("data-index", (x) => `${x["idx"]}`)
        .style("cursor", "pointer")
        .style("text-decoration", "line-through 3px solid #4682B4aa")
        .style("margin-right", "1px")
        .style("color", "black")
        .style("border-radius", "5px")
        .style("border-bottom", (x) => {
          const avgScore = x._children.reduce((acc, curr) => acc + curr[self.controls.colorBy], 0) / x._children.length;
          const color = colorScale(avgScore);
          return `5px solid ${color}`;
        })
        .html((x) => cleanSpan(x[self.controls.span]))
        .on("click", clickReplacedWord);
      synonymsWrappers.append("span").text("(");
      synonymsWrappers
        .append("span")
        .attr("data-index", (x) => `${x["idx"]}`)
        .attr("class", "synonyms-placeholder")
        .attr("hide", false)
        .text("...");
      synonymsWrappers
        .append("div")
        .attr("data-index", (x) => `${x["idx"]}`)
        .attr("class", "synonyms")
        .attr("hide", true) // hide by default
        .selectAll("div")
        .data((d) => {
          d._children.sort((a, b) => b[self.controls.colorBy] - a[self.controls.colorBy]);
          return d._children;
        })
        .join("div")
        .attr("class", "word")
        .style("color", "#4682B4")
        .style("margin-top", "2px")
        .style("border-top", "2px solid white")
        .style("border-radius", "5px")
        .style("border-bottom", (x) => {
          const color = colorScale(x[self.controls.colorBy]);
          return `5px solid ${color}`;
        })
        .html((x) => cleanSpan(x.synonym))
        .on("mouseenter", mouseenter)
        .on("mousemove", mousemove)
        .on("mouseleave", mouseleave);
      synonymsWrappers.append("span").text(")");
    } else {
      spanDivs
        .filter((x) => {
          const wordIsBreak = x.hasOwnProperty("type") && x.type == "break";
          return !wordIsBreak;
        })
        .style("margin-top", "2px")
        .style("border-top", "2px solid white")
        .style("border-radius", "5px")
        .style("border-bottom", (x) => {
          const color = colorScale(x[self.controls.colorBy]);
          return `5px solid ${color}`;
        })
        .on("mouseenter", mouseenter)
        .on("mousemove", mousemove)
        .on("mouseleave", mouseleave);
    }

    // ============================= HOVER ================================= //

    function mouseenter(event, d) {
      d3.select(this).style("border-top", "2px solid red").style("background-color", "#efefef");
      const selectedSummaryData = self.data[self.controls.summarySelectedID];
      const trueScore = selectedSummaryData.scores[self.controls.model];
      const trueLB = (Math.round(trueValueExtent[0] * 100) / 100).toFixed(2);
      const trueUB = (Math.round(trueValueExtent[1] * 100) / 100).toFixed(2);
      const absLB = (Math.round(absValueExtent[0] * 100) / 100).toFixed(2);
      const absUB = (Math.round(absValueExtent[1] * 100) / 100).toFixed(2);
      let html = `
        <div>True score:&nbsp;<b>${trueScore}</b></div>
        <div>New score:&nbsp;<b>${d.score}</b></div>
        <div>Diff (true) [${trueLB}, 0, ${trueUB}]:&nbsp;<b>${d.diff_true.toExponential()}</b></div>
        <div>Diff (abs) [${absLB}, ${absUB}]:&nbsp;<b>${d.diff_abs.toExponential()}</b></div>
      `;
      self.tooltip.html(html);
      self.tooltip.style("display", "block").style("opacity", 1);
    }
    function mousemove(event, d) {
      positionTooltip(event.x, event.y);
    }
    function mouseleave(event, d) {
      d3.select(this).style("border-top", "2px solid white").style("background-color", "");
      self.tooltip.html("");
      self.tooltip.style("display", "none").style("opacity", 0);
    }

    // ============================= CLICK ================================= //

    function clickReplacedWord(event, d) {
      const placeholder = d3.select(`#perturbation .synonyms-placeholder[data-index="${d["idx"]}"]`);
      const hidePlaceholder = placeholder.attr("hide") == "true" ? false : true;
      placeholder.attr("hide", hidePlaceholder);
      const synonyms = d3.select(`#perturbation .synonyms[data-index="${d["idx"]}"]`);
      const hideSynonyms = synonyms.attr("hide") == "true" ? false : true;
      synonyms.attr("hide", hideSynonyms);
    }

    // ============================= HELPER ================================ //

    const topPad = 12; // draw tooltip px up from top edge of cursor
    const leftPad = -12; // draw tooltip px left from left edge of cursor

    function positionTooltip(eventX, eventY) {
      const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
      const width = self.tooltip.node().getBoundingClientRect().width + 2;
      const height = self.tooltip.node().getBoundingClientRect().height + 2;
      const left = window.scrollX + eventX + width - leftPad >= vw ? vw - width : window.scrollX + eventX - leftPad;
      const top = window.scrollY + eventY - height - topPad <= 0 ? 0 : window.scrollY + eventY - height - topPad;
      self.tooltip.style("left", `${left}px`).style("top", `${top}px`);
    }

    function cleanSpan(span) {
      let newSpan = span;
      newSpan = self.utils.escapeHtml(newSpan); // escape HTML special characters and whitespace
      newSpan = self.utils.escapeEscapeSequences(newSpan); // escape JavaScript escape sequences
      return newSpan;
    }
  }
}
