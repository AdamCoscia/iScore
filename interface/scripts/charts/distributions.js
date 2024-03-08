export default class distributions {
  constructor(utils, elem) {
    this.utils = utils;
    this.svg = elem;
    this.node = elem.node();
    this.tooltip = d3.select("#charts-tooltip");
    this.layers = {
      interactions: null,
      drawing: null,
    };
    this.margin = {
      t: 4,
      r: 24,
      b: 24,
      l: 4,
    };
    this.data = null;
    this.controls = {
      render: false,
      xAttr: "slc",
      yAttr: "slw",
      convertAttr: {
        slc: "content",
        lcg: "content",
        slw: "wording",
        lwg: "wording",
      },
    };
  }

  init() {
    const self = this;

    const svg = self.svg;
    const margin = self.margin;

    // add groups in layer order (i.e., draw element groups in this order)
    const interactionsLayer = svg.append("g").attr("class", "interaction-layer");
    const drawingLayer = interactionsLayer.append("g").attr("class", "drawing-layer");

    // save groups to access later
    self.layers.interactions = interactionsLayer;
    self.layers.drawing = drawingLayer;
  }

  clear() {
    this.layers.drawing.selectAll("*").remove();
  }

  render() {
    const self = this;
    const show = document.getElementById("show-distributions-checkbox").checked;

    if (!self.controls.render || !show) return;

    const fw = parseFloat(self.svg.style("width"));
    const fh = parseFloat(self.svg.style("height"));

    const data = self.data.trainingData;

    // x-axis histogram parameters
    const xHist_g = self.layers.drawing.append("g").attr("class", "x-hist-g");
    const xHist_height = 40;
    const xHist_width = fw;
    const xHist_nBins = 40;
    const xHist_normalize = false;
    const xHist_color = "#2b2b2baa";

    // y-axis histogram parameters
    const yHist_g = self.layers.drawing
      .append("g")
      .attr("class", "y-hist-g")
      .attr(
        "transform",
        `
        rotate(-90, ${self.margin.l}, ${self.margin.t})
        translate(${-fh + self.margin.l + self.margin.t}, ${-self.margin.l + self.margin.t})
        `
      );
    const yHist_height = 40;
    const yHist_width = fh;
    const yHist_nBins = 40;
    const yHist_normalize = false;
    const yHist_color = "#2b2b2baa";

    // scatter plot parameters
    const scatter_g = self.layers.drawing.append("g").attr("class", "scatter-g");
    const scatter_r = 3; // (fixed) radius of dots; in pixels
    const scatter_fill = "#ffffff00"; // fill color for dots
    const scatter_stroke = "steelblue"; // stroke color for the dots
    const scatter_strokeWidth = 1.5; // stroke width for dots

    const mt = xHist_height + self.margin.t + 8;
    const mr = self.margin.r + 8;
    const mb = self.margin.b + 8;
    const ml = yHist_height + self.margin.l + 8;

    /* ===================================================================== */

    // draw top histogram of training data
    // See: <https://observablehq.com/@d3/histogram>

    // Compute values.
    const xHist_attr = self.controls.convertAttr[self.controls.xAttr];
    const xHist_X = d3.map(data, (x) => x[xHist_attr]);
    const xHist_Y0 = d3.map(data, () => 1);
    const xHist_I = d3.range(xHist_X.length);

    // Compute xHist_bins.
    const xHist_bins = d3
      .bin()
      .thresholds(xHist_nBins)
      .value((i) => xHist_X[i])(xHist_I);
    const xHist_Y = Array.from(xHist_bins, (xHist_I) => d3.sum(xHist_I, (i) => xHist_Y0[i]));
    if (xHist_normalize) {
      const total = d3.sum(xHist_Y);
      for (let i = 0; i < xHist_Y.length; ++i) xHist_Y[i] /= total;
    }

    // Compute default domains.
    const xHist_xDomain = [xHist_bins[0].x0, xHist_bins[xHist_bins.length - 1].x1];
    const xHist_xRange = [ml, xHist_width - mr];
    const xHist_yDomain = [0, d3.max(xHist_Y)];
    const xHist_yRange = [xHist_height + self.margin.t, self.margin.t];

    // Construct scales and axes.
    let xHist_xFormat = undefined;
    const xHist_xScale = d3.scaleLinear(xHist_xDomain, xHist_xRange);
    const xHist_yScale = d3.scaleLinear(xHist_yDomain, xHist_yRange);
    const xHist_xAxis = d3
      .axisBottom(xHist_xScale)
      .ticks(xHist_width / 80, xHist_xFormat)
      .tickSizeOuter(0);

    // save index of bin to itself
    xHist_bins.forEach((x, i) => (x["i"] = i));

    xHist_g
      .append("g")
      .attr("fill", xHist_color)
      .selectAll("rect")
      .data(xHist_bins)
      .join("rect")
      .attr("x", (d) => xHist_xScale(d.x0) + 0.5)
      .attr("width", (d) => Math.max(0, xHist_xScale(d.x1) - xHist_xScale(d.x0) - 1))
      .attr("y", (d, i) => xHist_yScale(xHist_Y[i]))
      .attr("height", (d, i) => xHist_yScale(0) - xHist_yScale(xHist_Y[i]))
      .on("mouseenter", xHist_mouseenter)
      .on("mousemove", xHist_mousemove)
      .on("mouseleave", xHist_mouseleave);

    xHist_g
      .append("g")
      .attr("transform", `translate(0,${fh - self.margin.b})`)
      .call(xHist_xAxis);

    /* ===================================================================== */

    // draw left histogram
    // See: <https://observablehq.com/@d3/histogram>

    // Compute values.
    const yHist_attr = self.controls.convertAttr[self.controls.yAttr];
    const yHist_X = d3.map(data, (x) => x[yHist_attr]);
    const yHist_Y0 = d3.map(data, () => 1);
    const yHist_I = d3.range(yHist_X.length);

    // Compute yHist_bins.
    const yHist_bins = d3
      .bin()
      .thresholds(yHist_nBins)
      .value((i) => yHist_X[i])(yHist_I);
    const yHist_Y = Array.from(yHist_bins, (yHist_I) => d3.sum(yHist_I, (i) => yHist_Y0[i]));
    if (yHist_normalize) {
      const total = d3.sum(yHist_Y);
      for (let i = 0; i < yHist_Y.length; ++i) yHist_Y[i] /= total;
    }

    // Compute default domains.
    const yHist_xDomain = [yHist_bins[0].x0, yHist_bins[yHist_bins.length - 1].x1];
    const yHist_xRange = [mb, yHist_width - mt];
    const yHist_yDomain = [0, d3.max(yHist_Y)];
    const yHist_yRange = [yHist_height + self.margin.l, self.margin.l];

    // Construct scales and axes.
    let yHist_xFormat = undefined;
    const yHist_xScale = d3.scaleLinear(yHist_xDomain, yHist_xRange);
    const yHist_yScale = d3.scaleLinear(yHist_yDomain, yHist_yRange);
    const yHist_xAxis = d3
      .axisBottom(yHist_xScale)
      .ticks(yHist_width / 80, yHist_xFormat)
      .tickSizeOuter(0);

    // save index of bin to itself
    yHist_bins.forEach((x, i) => (x["i"] = i));

    yHist_g
      .append("g")
      .attr("fill", yHist_color)
      .selectAll("rect")
      .data(yHist_bins)
      .join("rect")
      .attr("x", (d) => yHist_xScale(d.x0) + 0.5)
      .attr("width", (d) => Math.max(0, yHist_xScale(d.x1) - yHist_xScale(d.x0) - 1))
      .attr("y", (d, i) => yHist_yScale(yHist_Y[i]))
      .attr("height", (d, i) => yHist_yScale(0) - yHist_yScale(yHist_Y[i]))
      .on("mouseenter", yHist_mouseenter)
      .on("mousemove", yHist_mousemove)
      .on("mouseleave", yHist_mouseleave);

    yHist_g
      .append("g")
      .attr("transform", `translate(0,${fw - self.margin.r})`)
      .call(yHist_xAxis);

    /* ===================================================================== */

    // draw scatter plot of training data
    // See: <https://observablehq.com/@d3/scatterplot>

    const scatter_xAttr = self.controls.convertAttr[self.controls.xAttr];
    const scatter_yAttr = self.controls.convertAttr[self.controls.yAttr];
    const scatter_X = d3.map(data, (x) => x[scatter_xAttr]);
    const scatter_Y = d3.map(data, (y) => y[scatter_yAttr]);
    const scatter_I = d3.range(scatter_X.length).filter((i) => !isNaN(scatter_X[i]) && !isNaN(scatter_Y[i]));

    // Compute default domains.
    const scatter_xDomain = d3.extent(scatter_X);
    const scatter_xRange = [ml, fw - mr]; // [left; right]
    const scatter_yDomain = d3.extent(scatter_Y);
    const scatter_yRange = [fh - mb, mt]; // [bottom; top]

    // Construct scales and axes.
    const scatter_xScale = d3.scaleLinear(scatter_xDomain, scatter_xRange);
    const scatter_yScale = d3.scaleLinear(scatter_yDomain, scatter_yRange);

    const scatter_trainingG = scatter_g.append("g").attr("class", "training-circles");

    scatter_trainingG
      .selectAll("circle")
      .data(scatter_I)
      .join("circle")
      .attr("data-index", (i) => i)
      .attr("class", "training")
      .attr("fill", scatter_fill)
      .attr("stroke", scatter_stroke)
      .attr("stroke-width", scatter_strokeWidth)
      .attr("opacity", 0.1)
      .attr("cx", (i) => scatter_xScale(scatter_X[i]))
      .attr("cy", (i) => scatter_yScale(scatter_Y[i]))
      .attr("r", scatter_r)
      .on("mouseenter", scatter_mouseenter)
      .on("mousemove", scatter_mousemove)
      .on("mouseleave", scatter_mouseleave)
      .on("click", scatter_click);

    /* ===================================================================== */

    // plot current scores (if available)
    const scatter_scoredG = scatter_g.append("g").attr("class", "scored-circles");
    if (self.data.hasOwnProperty("sessionData")) {
      const totalRuns = self.data.totalRuns;
      let opacityScale = () => 1;
      if (totalRuns > 1) {
        opacityScale = d3.scaleLinear([totalRuns, 1], [1, 0.75]);
      }

      self.data.sessionData.forEach((p, i) => {
        const g = scatter_scoredG.append("g").attr("opacity", opacityScale(p.run));
        g.append("circle")
          .attr("data-index", i)
          .attr("data-id", p.id)
          .attr("data-run", p.run)
          .attr("class", "scored")
          .attr("fill", "goldenrod")
          .attr("cx", () => {
            let cx = ml - 8;
            if (p.scores.hasOwnProperty(self.controls.xAttr)) {
              cx = scatter_xScale(p.scores[self.controls.xAttr]);
            }
            return cx;
          })
          .attr("cy", () => {
            let cy = mt - 8;
            if (p.scores.hasOwnProperty(self.controls.yAttr)) {
              cy = scatter_yScale(p.scores[self.controls.yAttr]);
            }
            return cy;
          })
          .attr("r", 6)
          .on("mouseenter", scatter_mouseenter)
          .on("mousemove", scatter_mousemove)
          .on("mouseleave", scatter_mouseleave)
          .on("click", scatter_click);
        g.append("text")
          .attr("data-index", i)
          .attr("data-id", p.id)
          .attr("data-run", p.run)
          .attr("class", "scored")
          .attr("pointer-events", "none")
          .attr("font-size", "8px")
          .attr("stroke", "maroon")
          .attr("stroke-width", 1)
          .attr("dominant-baseline", "middle")
          .attr("text-anchor", "middle")
          .attr("x", () => {
            let x = ml - 8;
            if (p.scores.hasOwnProperty(self.controls.xAttr)) {
              x = scatter_xScale(p.scores[self.controls.xAttr]);
            }
            return x;
          })
          .attr("y", () => {
            let y = mt - 8;
            if (p.scores.hasOwnProperty(self.controls.yAttr)) {
              y = scatter_yScale(p.scores[self.controls.yAttr]);
            }
            return y;
          })
          .html(p.id);
      });

      scatter_scoredG.selectAll("circle.scored").each(function () {
        const p_from = d3.select(this);
        const i = parseInt(p_from.attr("data-id"));
        const r = parseInt(p_from.attr("data-run"));
        const p_to = scatter_scoredG.select(`circle.scored[data-id="${i}"][data-run="${r + 1}"]`);
        if (!p_to.empty()) {
          scatter_scoredG
            .append("line")
            .attr("pointer-events", "none")
            .attr("stroke", "black")
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "2 2")
            .attr("marker-end", "url(#arrow)")
            .attr("x1", parseInt(p_from.attr("cx")))
            .attr("x2", parseInt(p_to.attr("cx")))
            .attr("y1", parseInt(p_from.attr("cy")))
            .attr("y2", parseInt(p_to.attr("cy")))
            .lower();
        }
      });
    }

    // ============================= HOVER ================================= //

    function xHist_mouseenter(event, d) {
      let html = `Range: [${d.x0} ≤ x < ${d.x1}]<br />Count: ${xHist_Y[d.i]}`;
      self.tooltip.html(html);
      self.tooltip.style("display", "block").style("opacity", 1);
    }
    function xHist_mousemove(event, d) {
      positionTooltip(event.x, event.y);
    }
    function xHist_mouseleave(event, d) {
      self.tooltip.html("");
      self.tooltip.style("display", "none").style("opacity", 0);
    }

    function yHist_mouseenter(event, d) {
      let html = `Range: [${d.x0} ≤ x < ${d.x1}]<br />Value: ${yHist_Y[d.i]}`;
      self.tooltip.html(html);
      self.tooltip.style("display", "block").style("opacity", 1);
    }
    function yHist_mousemove(event, d) {
      positionTooltip(event.x, event.y);
    }
    function yHist_mouseleave(event, d) {
      self.tooltip.html("");
      self.tooltip.style("display", "none").style("opacity", 0);
    }

    function scatter_mouseenter(event, d) {
      const i = parseInt(d3.select(this).attr("data-index"));
      const p = typeof d == "undefined" ? self.data.sessionData[i] : data[i];
      const summary = p.summary.length > 80 ? p.summary.substring(0, 80) + "..." : p.summary;
      const source = p.source.length > 80 ? p.source.substring(0, 80) + "..." : p.source;
      let html = `
        <b><u>Summary</u></b>: ${self.utils.escapeHtml(summary)}<br />
        <b><u>Source</u></b>: ${self.utils.escapeHtml(source)}<br />
      `;
      if (typeof d == "undefined") {
        html += Object.entries(p.scores)
          .map(([model, score]) => `<b><u>${model}</u></b>: ${score}<br />`)
          .join("");
      } else {
        html += `
          <b><u>Content (PCA, normalized)</u></b>: ${p.content}<br />
          <b><u>Wording (PCA, normalized)</u></b>: ${p.wording}<br />
        `;
      }
      self.tooltip.html(html);
      self.tooltip.style("display", "block").style("opacity", 1);
    }
    function scatter_mousemove(event, d) {
      positionTooltip(event.x, event.y);
    }
    function scatter_mouseleave(event, d) {
      self.tooltip.html("");
      self.tooltip.style("display", "none").style("opacity", 0);
    }

    // ============================= CLICK ================================= //

    function scatter_click(event, d) {
      scatter_trainingG
        .selectAll("circle.training")
        .attr("fill", scatter_fill)
        .attr("stroke", scatter_stroke)
        .attr("stroke-width", scatter_strokeWidth)
        .attr("opacity", 0.1)
        .attr("r", scatter_r);
      scatter_scoredG
        .selectAll("circle.scored")
        .attr("fill", "goldenrod")
        .attr("stroke", "none")
        .attr("stroke-width", 0)
        .attr("r", 6);
      d3.select(this).attr("stroke", "maroon").attr("stroke-width", 2).attr("opacity", 1);

      const i = parseInt(d3.select(this).attr("data-index"));
      const p = typeof d == "undefined" ? self.data.sessionData[i] : data[i];

      const perturbationWrapper = document.getElementById("perturbation-wrapper");
      const attentionWrapper = document.getElementById("attention-wrapper");

      if (typeof d == "undefined") {
        perturbationWrapper.style.opacity = 1;
        perturbationWrapper.style.pointerEvents = "all";
        attentionWrapper.style.opacity = 1;
        attentionWrapper.style.pointerEvents = "all";
        document.getElementById("summary-select").value = p.id;
      } else {
        perturbationWrapper.style.opacity = 0.25;
        perturbationWrapper.style.pointerEvents = "none";
        attentionWrapper.style.opacity = 0.25;
        attentionWrapper.style.pointerEvents = "none";
      }

      // const sourceKeyphrases = p.hasOwnProperty("sourceKeyphrases") ? p.sourceKeyphrases : null;
      // const sourceKeyphrasesHTML = sourceKeyphrases !== null ? getKeyphrasesHTML(sourceKeyphrases) : "...";
      document.getElementById("source-input").value = p.source;
      document.getElementById("source-input").classList.remove("input-error");
      document.getElementById("examples-select").value = "unset";
      // document.getElementById("source-keyphrases").innerHTML = sourceKeyphrasesHTML;

      if (typeof d == "undefined") {
        document.getElementById(`summary-input-${p.id}`).value = p.summary;
        document.getElementById(`summary-input-${p.id}`).classList.remove("input-error");
      } else {
        document.getElementById(`summary-input-1`).value = p.summary;
        document.getElementById(`summary-input-1`).classList.remove("input-error");
      }
      document.getElementById("examples-select").value = "unset";
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

    function getKeyphrasesHTML(text) {
      return text
        .split(";")
        .filter((x) => x.trim().length > 0)
        .map((x) => `<div>${x.trim()}</div>`)
        .join("");
    }
  }
}
