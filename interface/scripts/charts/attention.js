export default class attention {
  constructor(utils, elem) {
    this.utils = utils;
    this.div = elem;
    this.node = elem.node();
    this.tooltip = d3.select("#charts-tooltip");
    this.layers = {
      interactions: null,
      drawing: null,
      attentionLayerOverview: null,
      attentionHeadOverview: null,
      attentionTokenOverview: null,
    };
    this.data = {};
    this.controls = {
      summarySelectedID: "1",
      render: false,
      model: "",
      layer: "",
      head: "",
      selectedToken: "",
    };
  }

  init() {
    const self = this;

    const div = self.div;
    const attentionLayerOverview = d3.select("#attention-layer-overview");
    const attentionHeadOverview = d3.select("#attention-head-overview");
    const attentionTokenOverview = d3.select("#attention-token-overview");

    // add groups in layer order (i.e., draw element groups in this order)
    const interactionsLayer = div.append("div").attr("class", "interaction-layer");
    const drawingLayer = interactionsLayer.append("div").attr("class", "drawing-layer");

    // save groups to access later
    self.layers.interactions = interactionsLayer;
    self.layers.drawing = drawingLayer;
    self.layers.attentionLayerOverview = attentionLayerOverview;
    self.layers.attentionHeadOverview = attentionHeadOverview;
    self.layers.attentionTokenOverview = attentionTokenOverview;
  }

  clear() {
    this.layers.drawing.selectAll("*").remove();
    this.layers.attentionLayerOverview.style("height", 0);
    this.layers.attentionLayerOverview.selectAll("*").remove();
    this.layers.attentionHeadOverview.style("height", 0);
    this.layers.attentionHeadOverview.selectAll("*").remove();
    this.layers.attentionTokenOverview.style("height", 0);
    this.layers.attentionTokenOverview.selectAll("*").remove();
  }

  render() {
    const self = this;
    const show = document.getElementById("show-attention-checkbox").checked;

    if (!self.controls.render || !show) return;

    const selectedSummaryData = self.data[self.controls.summarySelectedID];

    // localAttentions is the weight from a token to all other tokens without
    // global attention in the sequence. The weight from a token to a token
    // with global attention is 0 and should be accessed from the first x
    // attention weights

    // globalAttentions is the weight from a token with global attention to
    // all other tokens in the sequence

    // localShape == [layers, heads, seq_len, x + attention_window + 1]
    // globalShape == [layers, heads, seq_len, x]

    const localShape = selectedSummaryData.attentionsShapes[self.controls.model].local;
    const globalShape = selectedSummaryData.attentionsShapes[self.controls.model].global;
    const localAttentions = selectedSummaryData.attentions[self.controls.model].local;
    const globalAttentions = selectedSummaryData.attentions[self.controls.model].global;

    // get tokens
    const globalTokens = selectedSummaryData.tokens[self.controls.model].global; // indices of tokens with global attention
    const tokensRaw = selectedSummaryData.tokens[self.controls.model].all;
    const tokens = tokensRaw.map((x) => [].concat(...x, null)); // add another index to each token tuple

    // get array shapes
    const localNLayers = localShape[0]; // number of layers in local attention
    const localNHeads = localShape[1]; // number of heads in local attention
    const localSeqLen = localShape[2]; // sequence length in local attention
    const localXAttentionWindow1 = localShape[3]; // size of attention window plus global tokens (x) plus 1

    const globalNLayers = globalShape[0]; // number of layers in global attention
    const globalNHeads = globalShape[1]; // number of heads in global attention
    const globalSeqLen = globalShape[2]; // sequence length in global attention
    const globalX = globalShape[3]; // number of tokens with global attention (x)

    // get attention window
    const attentionWindow = localXAttentionWindow1 - globalX - 1;
    document.querySelector("#attention-window-span").innerHTML = `${attentionWindow} tokens`;

    // create color scale
    const colorScale = (scale) => d3.scaleSequential((x) => d3.interpolatePurples(scale(x)));

    // draw tokens
    const tokenDivs = self.layers.drawing
      .selectAll("div")
      .data(tokens)
      .join("div")
      .attr("data-index", (x) => x[0]) // make index available from DOM
      .attr("class", "token")
      .style("color", (x) => (globalTokens.includes(x[0]) ? "darkorange" : "black"))
      .style("font-weight", (x) => (globalTokens.includes(x[0]) ? "900" : "400"))
      .style("margin-top", "2px")
      .style("border-top", "2px solid white")
      .style("border-bottom", "5px solid white")
      .style("border-radius", "5px")
      .html((x) => cleanSpan(x[2]))
      .on("mouseenter", mouseenter)
      .on("mousemove", mousemove)
      .on("mouseleave", mouseleave)
      .on("click", click);

    const selectedToken = parseInt(self.controls.selectedToken);
    drawAttentionScoresAroundToken(selectedToken);
    if (document.getElementById("attention-heatmaps-checkbox").checked) {
      drawAttentionOverviews(selectedToken);
    }

    // ============================= HOVER ================================= //

    function mouseenter(event, d) {
      if (d[3] !== null) {
        d3.select(this).style("border-top", "2px solid red").style("background-color", "#efefef");
        const t = tokens[parseInt(self.controls.selectedToken)];
        const from = cleanSpan(t[2]);
        const to = cleanSpan(d[2]);
        const score = d[3].toExponential();
        let html = `Attention &#x2022; ${from} &#187; ${to} &#x2022; ${score}`;
        self.tooltip.html(html);
        self.tooltip.style("display", "block").style("opacity", 1);
      }
    }
    function mousemove(event, d) {
      if (d[3] !== null) {
        positionTooltip(event.x, event.y);
      }
    }
    function mouseleave(event, d) {
      if (d[3] !== null) {
        d3.select(this).style("border-top", "2px solid white").style("background-color", "");
        self.tooltip.html("");
        self.tooltip.style("display", "none").style("opacity", 0);
      }
    }

    // ============================= CLICK ================================= //

    function click(event, d) {
      self.controls.selectedToken = `${d[0]}`; // store as string
      document.getElementById("attention-token-slider").value = d[0];
      drawAttentionScoresAroundToken(d[0]);
      if (document.getElementById("attention-heatmaps-checkbox").checked) {
        drawAttentionOverviews(d[0]);
      }
    }

    // ============================= HELPER ================================ //

    function drawAttentionOverviews(tokenFrom) {
      const currLayer = parseInt(self.controls.layer) - 1;
      const currHead = parseInt(self.controls.head) - 1;
      const tokenFromIsGlobal = globalTokens.includes(tokenFrom);

      self.layers.attentionLayerOverview.style("height", "48px");
      self.layers.attentionLayerOverview.selectAll("*").remove();
      self.layers.attentionHeadOverview.style("height", "48px");
      self.layers.attentionHeadOverview.selectAll("*").remove();

      let alo_xDomain = [];
      let aho_xDomain = [];
      if (!tokenFromIsGlobal) {
        alo_xDomain = [...Array(localNLayers).keys()];
        aho_xDomain = [...Array(localNHeads).keys()];
      } else {
        alo_xDomain = [...Array(globalNLayers).keys()];
        aho_xDomain = [...Array(globalNHeads).keys()];
      }

      const alo_yDomain = [...Array(tokens.length).keys()];
      const aho_yDomain = [...Array(tokens.length).keys()];

      const alo_data = [];
      const alo_fw = parseFloat(self.layers.attentionLayerOverview.style("width"));
      const alo_fh = parseFloat(self.layers.attentionLayerOverview.style("height"));
      const alo_xScale = d3.scaleBand().domain(alo_xDomain).range([0, alo_fw]);
      const alo_yScale = d3.scaleBand().domain(alo_yDomain).range([0, alo_fh]);
      const alo_g = self.layers.attentionLayerOverview.append("g");

      const aho_data = [];
      const aho_fw = parseFloat(self.layers.attentionHeadOverview.style("width"));
      const aho_fh = parseFloat(self.layers.attentionHeadOverview.style("height"));
      const aho_xScale = d3.scaleBand().domain(aho_xDomain).range([0, aho_fw]);
      const aho_yScale = d3.scaleBand().domain(aho_yDomain).range([0, aho_fh]);
      const aho_g = self.layers.attentionHeadOverview.append("g");

      if (!tokenFromIsGlobal) {
        const halfAttentionWindow = Math.floor(attentionWindow / 2);

        // get overview at currHead for every layer
        for (let layer = 0; layer < localNLayers; layer++) {
          const i = layer * localNLayers * localSeqLen + currHead * localSeqLen + tokenFrom;
          const attentions = localAttentions[i];
          const tokenToLB = Math.max(globalX, tokenFrom - halfAttentionWindow);
          const tokenToUB = Math.min(localSeqLen - 1, tokenFrom + halfAttentionWindow);

          // create color scale
          const jLB = halfAttentionWindow + globalX + tokenToLB - tokenFrom;
          const jUB = halfAttentionWindow + globalX + tokenToUB - tokenFrom;
          const ls = attentions.slice(jLB, jUB + 1).filter((x) => x > 0); // get local scores, removing zeros
          const gs = attentions.slice(0, globalX).filter((x) => x > 0); // get global scores, removing zeros
          const scores = ls.concat(gs);
          const logScale = d3.scaleLog().domain(d3.extent(scores));
          const colorScaleLog = colorScale(logScale);

          // if drawing from token to token with global attention, access that
          // value from first x attention weights. Otherwise, attention is
          // relative distance b/w tokenFrom and tokenTo, starting in the
          // center of the rest of the array at halfAttentionWindow + globalX.

          let j, attnScore, color;
          for (let tokenTo = tokenToLB; tokenTo <= tokenToUB; tokenTo++) {
            j = halfAttentionWindow + globalX + tokenTo - tokenFrom;
            attnScore = attentions[j];
            color = attnScore == 0 ? "red" : colorScaleLog(attnScore);
            alo_data.push({ layer: layer, head: currHead, token: tokenTo, score: attnScore, color: color });
          }
          globalTokens.forEach((globalTokenTo) => {
            j = globalTokens.indexOf(globalTokenTo);
            attnScore = attentions[j];
            color = attnScore == 0 ? "red" : colorScaleLog(attnScore);
            alo_data.push({ layer: layer, head: currHead, token: globalTokenTo, score: attnScore, color: color });
          });
        }

        // get overview of currLayer for every head
        for (let head = 0; head < localNHeads; head++) {
          const i = currLayer * localNLayers * localSeqLen + head * localSeqLen + tokenFrom;
          const attentions = localAttentions[i];
          const tokenToLB = Math.max(globalX, tokenFrom - halfAttentionWindow);
          const tokenToUB = Math.min(localSeqLen - 1, tokenFrom + halfAttentionWindow);

          // create color scale
          const jLB = halfAttentionWindow + globalX + tokenToLB - tokenFrom;
          const jUB = halfAttentionWindow + globalX + tokenToUB - tokenFrom;
          const ls = attentions.slice(jLB, jUB + 1).filter((x) => x > 0); // get local scores, removing zeros
          const gs = attentions.slice(0, globalX).filter((x) => x > 0); // get global scores, removing zeros
          const scores = ls.concat(gs);
          const logScale = d3.scaleLog().domain(d3.extent(scores));
          const colorScaleLog = colorScale(logScale);

          // if drawing from token to token with global attention, access that
          // value from first x attention weights. Otherwise, attention is
          // relative distance b/w tokenFrom and tokenTo, starting in the
          // center of the rest of the array at halfAttentionWindow + globalX.

          let j, attnScore, color;
          for (let tokenTo = tokenToLB; tokenTo <= tokenToUB; tokenTo++) {
            j = halfAttentionWindow + globalX + tokenTo - tokenFrom;
            attnScore = attentions[j];
            color = attnScore == 0 ? "red" : colorScaleLog(attnScore);
            aho_data.push({ layer: currLayer, head: head, token: tokenTo, score: attnScore, color: color });
          }
          globalTokens.forEach((globalTokenTo) => {
            j = globalTokens.indexOf(globalTokenTo);
            attnScore = attentions[j];
            color = attnScore == 0 ? "red" : colorScaleLog(attnScore);
            aho_data.push({ layer: currLayer, head: head, token: globalTokenTo, score: attnScore, color: color });
          });
        }
      }

      if (tokenFromIsGlobal) {
        // get overview at currHead for every layer
        for (let layer = 0; layer < localNLayers; layer++) {
          const startIndex = layer * globalNLayers * globalSeqLen + currHead * globalSeqLen;

          // create color scale
          const scores = [];
          globalAttentions.slice(startIndex, startIndex + globalSeqLen).forEach((a) => {
            a.forEach((b) => {
              if (b > 0) scores.push(b);
            });
          });
          const logScale = d3.scaleLog().domain(d3.extent(scores));
          const colorScaleLog = colorScale(logScale);

          // if drawing from token with global attention to token, access that
          // value from the global attention weights

          let i, tokenTo, attnScore, color;
          const j = globalTokens.indexOf(tokenFrom);
          tokens.forEach((x) => {
            tokenTo = x[0];
            i = startIndex + tokenTo;
            attnScore = globalAttentions[i][j];
            color = attnScore == 0 ? "red" : colorScaleLog(attnScore);
            alo_data.push({ layer: layer, head: currHead, token: tokenTo, score: attnScore, color: color });
          });
        }

        // get overview at currLayer for every head
        for (let head = 0; head < localNHeads; head++) {
          const startIndex = currLayer * globalNLayers * globalSeqLen + head * globalSeqLen;

          // create color scale
          const scores = [];
          globalAttentions.slice(startIndex, startIndex + globalSeqLen).forEach((a) => {
            a.forEach((b) => {
              if (b > 0) scores.push(b);
            });
          });
          const logScale = d3.scaleLog().domain(d3.extent(scores));
          const colorScaleLog = colorScale(logScale);

          // if drawing from token with global attention to token, access that
          // value from the global attention weights

          let i, tokenTo, attnScore, color;
          const j = globalTokens.indexOf(tokenFrom);
          tokens.forEach((x) => {
            tokenTo = x[0];
            i = startIndex + tokenTo;
            attnScore = globalAttentions[i][j];
            color = attnScore == 0 ? "red" : colorScaleLog(attnScore);
            aho_data.push({ layer: currLayer, head: head, token: tokenTo, score: attnScore, color: color });
          });
        }
      }

      // draw overviews
      alo_g
        .selectAll("rect")
        .data(alo_data)
        .join("rect")
        .attr("x", (d) => alo_xScale(d.layer))
        .attr("y", (d) => alo_yScale(d.token))
        .attr("width", alo_xScale.bandwidth())
        .attr("height", alo_yScale.bandwidth())
        .attr("fill", (d) => d.color);
      aho_g
        .selectAll("rect")
        .data(aho_data)
        .join("rect")
        .attr("x", (d) => aho_xScale(d.head))
        .attr("y", (d) => aho_yScale(d.token))
        .attr("width", aho_xScale.bandwidth())
        .attr("height", aho_yScale.bandwidth())
        .attr("fill", (d) => d.color);
    }

    function drawAttentionScoresAroundToken(tokenFrom) {
      d3.selectAll("#attention .token")
        .style("color", (x) => (globalTokens.includes(x[0]) ? "darkorange" : "black"))
        .style("font-weight", (x) => (globalTokens.includes(x[0]) ? "900" : "400"))
        .style("border-bottom", "5px solid white"); // clear prev borders
      tokenDivs.each((x) => (x[3] = null)); // reset scores

      d3.select(`#attention .token[data-index='${tokenFrom}']`).style("color", "crimson").style("font-weight", "900");

      const ato_data = [];

      const layer = parseInt(self.controls.layer) - 1;
      const head = parseInt(self.controls.head) - 1;
      const tokenFromIsGlobal = globalTokens.includes(tokenFrom);

      if (!tokenFromIsGlobal) {
        const halfAttentionWindow = Math.floor(attentionWindow / 2);

        const i = layer * localNLayers * localSeqLen + head * localSeqLen + tokenFrom;
        const attentions = localAttentions[i];
        const tokenToLB = Math.max(globalX, tokenFrom - halfAttentionWindow);
        const tokenToUB = Math.min(localSeqLen - 1, tokenFrom + halfAttentionWindow);

        // create color scale
        const jLB = halfAttentionWindow + globalX + tokenToLB - tokenFrom;
        const jUB = halfAttentionWindow + globalX + tokenToUB - tokenFrom;
        const ls = attentions.slice(jLB, jUB + 1).filter((x) => x > 0); // get local scores, removing zeros
        const gs = attentions.slice(0, globalX).filter((x) => x > 0); // get global scores, removing zeros
        const scores = ls.concat(gs);
        const logScale = d3.scaleLog().domain(d3.extent(scores));
        const colorScaleLog = colorScale(logScale);

        // if drawing from token to token with global attention, access that
        // value from first x attention weights. Otherwise, attention is
        // relative distance b/w tokenFrom and tokenTo, starting in the
        // center of the rest of the array at halfAttentionWindow + globalX.

        let j, attnScore, color;
        globalTokens.forEach((globalTokenTo) => {
          j = globalTokens.indexOf(globalTokenTo);
          attnScore = attentions[j];
          tokenDivs
            .filter((x) => x[0] === globalTokenTo)
            .style("border-bottom", (x) => {
              x[3] = attnScore; // save attention score
              color = attnScore == 0 ? "red" : colorScaleLog(attnScore);
              ato_data.push({ layer: layer, head: head, token: globalTokenTo, score: attnScore, color: color });
              return `5px solid ${color}`;
            });
        });
        for (let tokenTo = tokenToLB; tokenTo <= tokenToUB; tokenTo++) {
          j = halfAttentionWindow + globalX + tokenTo - tokenFrom;
          attnScore = attentions[j];
          tokenDivs
            .filter((x) => x[0] === tokenTo)
            .style("border-bottom", (x) => {
              x[3] = attnScore; // save attention score
              color = attnScore == 0 ? "red" : colorScaleLog(attnScore);
              ato_data.push({ layer: layer, head: head, token: tokenTo, score: attnScore, color: color });
              return `5px solid ${color}`;
            });
        }
      }

      if (tokenFromIsGlobal) {
        const startIndex = layer * globalNLayers * globalSeqLen + head * globalSeqLen;

        // create color scale
        const scores = [];
        globalAttentions.slice(startIndex, startIndex + globalSeqLen).forEach((a) => {
          a.forEach((b) => {
            if (b > 0) scores.push(b);
          });
        });
        const logScale = d3.scaleLog().domain(d3.extent(scores));
        const colorScaleLog = colorScale(logScale);

        // if drawing from token with global attention to token, access that
        // value from the global attention weights

        let i, tokenTo, attnScore, color;
        const j = globalTokens.indexOf(tokenFrom);
        tokenDivs.style("border-bottom", (x) => {
          tokenTo = x[0];
          i = startIndex + tokenTo;
          attnScore = globalAttentions[i][j];
          x[3] = attnScore; // save attention score
          color = attnScore == 0 ? "red" : colorScaleLog(attnScore);
          ato_data.push({ layer: layer, head: head, token: tokenTo, score: attnScore, color: color });
          return `5px solid ${color}`;
        });
      }

      if (document.getElementById("attention-heatmaps-checkbox").checked) {
        // draw overview
        self.layers.attentionTokenOverview.style("height", "8px");
        self.layers.attentionTokenOverview.selectAll("*").remove();
        const ato_xDomain = [...Array(tokens.length).keys()];
        const ato_fw = parseFloat(self.layers.attentionTokenOverview.style("width"));
        const ato_fh = parseFloat(self.layers.attentionTokenOverview.style("height"));
        const ato_xScale = d3.scaleBand().domain(ato_xDomain).range([0, ato_fw]);
        const ato_g = self.layers.attentionTokenOverview.append("g");
        ato_g
          .selectAll("rect")
          .data(ato_data)
          .join("rect")
          .attr("x", (d) => ato_xScale(d.token))
          .attr("y", 0)
          .attr("width", ato_xScale.bandwidth())
          .attr("height", `${ato_fh}px`)
          .attr("fill", (d) => d.color);
      }
    }

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
