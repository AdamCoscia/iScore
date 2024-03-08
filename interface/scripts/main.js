import utils from "./utils.js";
import Store from "./store.js";

import Examples from "../assets/data/examples.js";
import TrainingData from "../assets/data/trainingData.js";

import Distributions from "./charts/distributions.js";
import Perturbation from "./charts/perturbation.js";
import Attention from "./charts/attention.js";

/**
 * See: <https://stackoverflow.com/a/17368101>
 * @param {*} other_array
 */
Array.prototype.extend = function (other_array) {
  other_array.forEach(function (v) {
    this.push(v);
  }, this);
};

// globals
var store = new Store({
  utils,
  Examples,
  TrainingData,
  Distributions,
  Perturbation,
  Attention,
});

// fire events when the page loads
window.onload = init;

function init() {
  initElements();
  initCharts();
  setEventListeners();
  store.charts.distributions.data = { trainingData: store.data.trainingData };
  store.charts.distributions.controls.render = true;
  store.charts.distributions.render();
}

function initElements() {
  // create elements for each example
  const examplesOpts = [];
  const opt = document.createElement("option");
  opt.value = "unset";
  opt.selected = true;
  opt.disabled = true;
  opt.hidden = true;
  opt.innerHTML = "Choose";
  examplesOpts.push(opt);
  for (const exampleName of Object.keys(store.data.examples)) {
    const opt = document.createElement("option");
    opt.value = exampleName;
    opt.innerHTML = exampleName;
    examplesOpts.push(opt);
  }
  store.elements.examplesSelect.replaceChildren(...examplesOpts);

  // create elements for each model
  const nModelSelects = 4;
  const modelOpts = [...Array(nModelSelects)].map(() => Array(nModelSelects));
  const tableColumns = [
    {
      title: "ID",
      field: "id",
      visible: false,
      resizable: false,
    },
    {
      title: "Run",
      field: "run",
      resizable: false,
    },
  ];
  const modelCheckboxes = [];
  const models = Object.entries(store.models);
  for (let i = 0; i < models.length; i++) {
    const modelKey = models[i][0];
    const modelValues = models[i][1];
    for (let j = 0; j < nModelSelects; j++) {
      const opt = document.createElement("option");
      opt.value = modelKey;
      opt.innerHTML = modelValues.label;
      modelOpts[j][i] = opt;
    }
    const modelCheckboxTemplateString = `
      <div class="flex-row ai-center">
        <input type="checkbox" id="model-${modelKey}-checkbox" name="model-${modelKey}-checkbox" checked />
        <label for="model-${modelKey}-checkbox">${modelValues.label}</label>
      </div>`;
    const modelCheckboxTemplate = utils.createElementFromTemplate(modelCheckboxTemplateString);
    modelCheckboxes.push(modelCheckboxTemplate);
    const column = {
      title: modelValues.label,
      field: modelKey,
      resizable: false,
    };
    tableColumns.push(column);
  }
  store.elements.distributionsXSelect.replaceChildren(...modelOpts[0]);
  store.elements.distributionsYSelect.replaceChildren(...modelOpts[1]);
  store.elements.distributionsYSelect.options[2].selected = true;
  store.elements.perturbationModelSelect.replaceChildren(...modelOpts[2]);
  store.elements.attentionModelSelect.replaceChildren(...modelOpts[3]);
  store.elements.modelCheckboxes.replaceChildren(...modelCheckboxes);

  // instantiate tabulator table
  store.tables.scoresTable = new Tabulator("#scores-table", {
    data: [],
    layout: "fitData",
    groupBy: "id",
    groupHeader: (value, count) => {
      const s = count > 1 ? "s" : "";
      return (
        "Summary ID: " + value + "<span style='color:#d00; margin-left:10px;'>(" + count + " item" + s + ")</span>"
      );
    },
    columns: tableColumns,
  });
}

function initCharts() {
  Object.values(store.charts).forEach((chart) => chart.init());
}

function requestData(requests) {
  updateCurrentCharts("clear");
  disableControls();

  const connect = fetch(store.backendURL.connect, {
    method: "GET",
    headers: {
      "Access-Control-Allow-Origin": "*",
    },
  });

  const getData = fetch(store.backendURL.getData, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      Connection: "keep-alive",
    },
    body: JSON.stringify(requests),
  });

  console.log("requesting data...");

  connect
    .then(() => {
      getData
        .then(processStreamData)
        .catch((error) => utils.handleFetchError(error))
        .finally(() => (store.elements.dataQueryButton.disabled = false));
    })
    .catch((error) => utils.handleFetchError(error));

  /**
   * Process streamed data from server delimited by `\n`
   * @param {*} response
   */
  async function processStreamData(response) {
    function readChunks(reader) {
      return {
        async *[Symbol.asyncIterator]() {
          let readResult = await reader.read();
          while (!readResult.done) {
            yield readResult.value;
            readResult = await reader.read();
          }
        },
      };
    }

    console.log("response recieved!");
    const textDecoder = new TextDecoder();
    const reader = response.body.getReader();

    console.log("reading chunks...");
    let chunkText = "";
    let objStrings = [];
    let prevObjString = "";
    let data = [];
    for await (const chunk of readChunks(reader)) {
      chunkText = prevObjString + textDecoder.decode(chunk);
      objStrings = chunkText.split("\n"); // server chunks objs by delimiting with newline character
      prevObjString = objStrings.pop(); // remove last string, as it may be cut off from next chunk
      data.extend(objStrings.map((x) => JSON.parse(x))); // convert strings to objects and merge into single array
    }

    console.log("chunks read!");
    saveData(data, requests);
  }
}

function saveData(data, requests) {
  const [perturbationData, attentionData] = processRequests(data, requests);

  const summaryID = store.data.summarySelectedID;
  const summaryOpts = store.data.summaryOptions[summaryID];

  // DISTRIBUTIONS

  const distributions = store.charts.distributions;
  distributions.data = {
    totalRuns: store.data.runCounter,
    sessionData: store.data.sessionData,
    trainingData: store.data.trainingData,
  };

  // PERTURBATIONS

  const perturbation = store.charts.perturbation;
  perturbation.controls.summarySelectedID = summaryID;
  perturbation.controls.render = summaryOpts.perturbation.render;
  perturbation.controls.model = summaryOpts.perturbation.model;
  perturbation.controls.span = summaryOpts.perturbation.span;
  perturbation.controls.colorBy = summaryOpts.perturbation.colorBy;
  perturbation.data = perturbationData;

  // ATTENTIONS

  const attention = store.charts.attention;
  attention.controls.summarySelectedID = summaryID;
  attention.controls.render = summaryOpts.attention.render;
  attention.controls.model = summaryOpts.attention.model;
  attention.controls.layer = summaryOpts.attention.layer;
  attention.controls.head = summaryOpts.attention.head;
  attention.controls.selectedToken = summaryOpts.attention.selectedToken;
  attention.data = attentionData;

  enableControls(summaryOpts);
  updateCurrentCharts("render");
  if (summaryOpts.attention.render) {
    updateAttentionSliders({
      layer: summaryOpts.attention.layer,
      head: summaryOpts.attention.head,
      token: summaryOpts.attention.selectedToken,
    });
  }

  store.data.runCounter++;
}

function disableControls() {
  store.elements.dataQueryButton.disabled = true;
  store.elements.summarySelect.disabled = true;

  // distributions
  store.elements.distributionsXSelect.disabled = true;
  store.elements.distributionsYSelect.disabled = true;
  store.elements.distributionsResetButton.disabled = true;

  // perturbation
  store.charts.perturbation.controls.render = false;

  // span
  const perturbationSpanSelect = store.elements.perturbationSpanSelect;
  perturbationSpanSelect.disabled = true;
  for (let i = 0; i < perturbationSpanSelect.length; i++) {
    perturbationSpanSelect.options[i].hidden = false;
  }
  perturbationSpanSelect.options.selectedIndex = 0;

  // model
  const perturbationModelSelect = store.elements.perturbationModelSelect;
  perturbationModelSelect.disabled = true;
  for (let i = 0; i < perturbationModelSelect.length; i++) {
    perturbationModelSelect.options[i].hidden = false;
  }
  perturbationModelSelect.options.selectedIndex = 0;

  // color by
  store.elements.perturbationColorBySelect.disabled = true;

  // attention
  store.charts.attention.controls.render = false;

  // model
  const attentionModelSelect = store.elements.attentionModelSelect;
  attentionModelSelect.disabled = true;
  for (let i = 0; i < attentionModelSelect.length; i++) {
    attentionModelSelect.options[i].hidden = false;
  }
  attentionModelSelect.options.selectedIndex = 0;

  // sliders
  store.elements.attentionHeatmapsCheckbox.disabled = true;
  store.elements.attentionLayerSlider.disabled = true;
  store.elements.attentionHeadSlider.disabled = true;
  store.elements.attentionTokenSlider.disabled = true;
  store.elements.attentionLayerTicks.replaceChildren();
  store.elements.attentionHeadTicks.replaceChildren();

  // window
  document.querySelector("#attention-window-span").innerHTML = "...";
}

function enableControls(opts) {
  store.elements.dataQueryButton.disabled = false;
  store.elements.summarySelect.disabled = false;

  // distributions
  store.elements.distributionsXSelect.disabled = false;
  store.elements.distributionsYSelect.disabled = false;
  store.elements.distributionsResetButton.disabled = false;

  // perturbation
  if (opts.perturbation.render) {
    // span
    const perturbationSpanSelect = store.elements.perturbationSpanSelect;
    perturbationSpanSelect.disabled = false;
    for (const [key, value] of Object.entries(opts.perturbation.enable)) {
      perturbationSpanSelect.querySelector(`option[value=${key}]`).hidden = !value;
    }
    perturbationSpanSelect.value = opts.perturbation.span;

    // model
    const perturbationModelSelect = store.elements.perturbationModelSelect;
    perturbationModelSelect.disabled = false;
    for (let i = 0; i < perturbationModelSelect.length; i++) {
      const opt = perturbationModelSelect.options[i];
      opt.hidden = !opts.models.includes(opt.value);
    }
    perturbationModelSelect.value = opts.perturbation.model;

    // color by
    store.elements.perturbationColorBySelect.disabled = false;
  }

  // attention
  if (opts.attention.render) {
    // model
    const attentionModelSelect = store.elements.attentionModelSelect;
    attentionModelSelect.disabled = false;
    for (let i = 0; i < attentionModelSelect.length; i++) {
      const opt = attentionModelSelect.options[i];
      opt.hidden = !opts.models.includes(opt.value);
    }
    attentionModelSelect.value = opts.attention.model;

    // sliders
    store.elements.attentionHeatmapsCheckbox.disabled = false;
    store.elements.attentionLayerSlider.disabled = false;
    store.elements.attentionHeadSlider.disabled = false;
    store.elements.attentionTokenSlider.disabled = false;
  }
}

function updateCurrentCharts(task) {
  const currentCharts = store.charts;
  Object.values(currentCharts).forEach((chart) => {
    const display = chart.node.getAttribute("display");
    if (display === null || display === "") {
      switch (task) {
        case "render":
          chart.render();
          break;
        case "clear":
          chart.clear();
          break;
      }
    }
  });
}

function setEventListeners() {
  // SHOW/HIDE TOGGLES

  store.elements.showScoresCheckbox.addEventListener("click", () => {
    hideWrappers(["scores-wrapper"]);
    if (store.elements.showScoresCheckbox.checked) {
      store.tables.scoresTable.replaceData(store.data.scoresTableData);
    }
  });
  store.elements.showDistributionsCheckbox.addEventListener("click", () => {
    hideWrappers(["distributions-wrapper"]);
    if (store.elements.showDistributionsCheckbox.checked) {
      store.charts.distributions.clear();
      store.charts.distributions.render();
    }
  });
  store.elements.showPerturbationCheckbox.addEventListener("click", () => {
    hideWrappers(["perturbation-wrapper"]);
    if (store.elements.showPerturbationCheckbox.checked) {
      store.charts.perturbation.clear();
      store.charts.perturbation.render();
    }
  });
  store.elements.showAttentionCheckbox.addEventListener("click", () => {
    hideWrappers(["attention-wrapper"]);
    if (store.elements.showAttentionCheckbox.checked) {
      store.charts.attention.clear();
      store.charts.attention.render();
    }
  });
  store.elements.attentionHeatmapsCheckbox.addEventListener("click", () => {
    hideWrappers(["attention-layer-overview", "attention-head-overview", "attention-token-overview"]);
    store.charts.attention.clear();
    store.charts.attention.render();
  });

  // QUERY OPTIONS

  store.elements.modelCheckboxesHide.addEventListener("click", () => {
    const modelCheckboxesHidden = store.elements.modelCheckboxes.style.display;
    if (modelCheckboxesHidden !== null && modelCheckboxesHidden == "none") {
      store.elements.modelCheckboxes.style.display = "block";
    } else {
      store.elements.modelCheckboxes.style.display = "none";
    }
  });

  store.elements.examplesSelect.addEventListener("change", () => {
    const exampleName = store.elements.examplesSelect.value;
    const source = store.data.examples[exampleName].source;
    const summary = store.data.examples[exampleName].summary;
    store.elements.sourceInput.value = source;
    store.elements.sourceInput.classList.remove("input-error");
    store.elements.summaryInput1.value = summary;
    store.elements.summaryInput1.classList.remove("input-error");
  });

  store.elements.dataQueryButton.addEventListener("click", () => {
    const sourceInput = store.elements.sourceInput;
    const summaryInputWrappers = store.elements.summaryInputContainer.querySelectorAll(".summary-input-wrapper");
    const modelCheckboxes = store.elements.modelCheckboxes;
    const run = store.data.runCounter;
    const requests = [];

    // assume all inputs are valid
    sourceInput.classList.remove("input-error");
    summaryInputWrappers.forEach((el) => el.querySelector(".summary-input").classList.remove("input-error"));
    modelCheckboxes.classList.remove("input-error");

    // valid model checkboxes
    const models = Object.keys(store.models).filter(
      (model) => document.getElementById(`model-${model}-checkbox`).checked
    );
    const validModels = models.length > 0;
    if (!validModels) {
      modelCheckboxes.classList.add("input-error");
      return;
    }

    // validate source
    const source = sourceInput.value;
    const validSource = source.length > 0;
    if (!validSource) {
      sourceInput.classList.add("input-error");
      return;
    }

    // validate summary
    for (const summaryInputWrapper of summaryInputWrappers) {
      const summaryInput = summaryInputWrapper.querySelector(".summary-input");
      const summary = summaryInput.value;
      const validSummary = summary.length > 0;
      if (!validSummary) {
        summaryInput.classList.add("input-error");
        return;
      }

      const id = parseInt(summaryInputWrapper.getAttribute("data-id"));
      const getGrammarPerturbation = summaryInputWrapper.querySelector(`#grammar-perturbation-checkbox-${id}`).checked
        ? "true"
        : "false";
      const getSentencePerturbation = summaryInputWrapper.querySelector(`#sentence-perturbation-checkbox-${id}`).checked
        ? "true"
        : "false";
      const getWordPerturbation = summaryInputWrapper.querySelector(`#word-perturbation-checkbox-${id}`).checked
        ? "true"
        : "false";
      const getTokenPerturbation = summaryInputWrapper.querySelector(`#token-perturbation-checkbox-${id}`).checked
        ? "true"
        : "false";
      const getAttentionScores = summaryInputWrapper.querySelector(`#get-attention-checkbox-${id}`).checked
        ? "true"
        : "false";

      requests.push({
        models,
        id,
        run,
        source,
        summary,
        getGrammarPerturbation,
        getSentencePerturbation,
        getWordPerturbation,
        getTokenPerturbation,
        getAttentionScores,
      });
    }

    // disable model analysis
    document.getElementById("attention-wrapper").style.opacity = 1;
    document.getElementById("attention-wrapper").style.pointerEvents = "all";
    document.getElementById("perturbation-wrapper").style.opacity = 1;
    document.getElementById("perturbation-wrapper").style.pointerEvents = "all";

    requestData(requests);
  });

  // SOURCE INPUT

  store.elements.sourceInput.addEventListener("input", () => {
    store.elements.sourceInput.classList.remove("input-error");
    store.elements.examplesSelect.value = "unset";
  });
  store.elements.sourceInput.addEventListener("keydown", function (e) {
    // Allow tab character to be inserted using tab key
    // See: <https://stackoverflow.com/a/6637396>
    if (e.key == "Tab") {
      e.preventDefault();
      var start = this.selectionStart;
      var end = this.selectionEnd;
      this.value = this.value.substring(0, start) + "\t" + this.value.substring(end);
      this.selectionStart = this.selectionEnd = start + 1;
    }
  });
  store.elements.sourceUploadButton.addEventListener("click", () => {
    store.elements.sourceFileUpload.click();
  });
  store.elements.sourceFileUpload.addEventListener("change", function () {
    const file = this.files[0];
    file.text().then((source) => {
      store.elements.sourceInput.value = source;
      store.elements.sourceInput.classList.remove("input-error");
      store.elements.examplesSelect.value = "unset";
    });
  });

  // SUMMARY INPUT

  store.elements.summaryUploadButton.addEventListener("click", () => {
    store.elements.summaryFileUpload.click();
  });
  store.elements.summaryFileUpload.addEventListener("change", function () {
    const file = this.files[0];
    file.text().then((summary) => {
      store.elements.summaryInput1.value = summary;
      store.elements.summaryInput1.classList.remove("input-error");
      store.elements.examplesSelect.value = "unset";
    });
  });

  store.elements.summaryInput1.addEventListener("input", () => {
    store.elements.summaryInput1.classList.remove("input-error");
    store.elements.examplesSelect.value = "unset";
  });
  store.elements.summaryInput1.addEventListener("keydown", function (e) {
    enableTabbing(e, this);
  });

  store.elements.summaryAddButton.addEventListener("click", () => {
    store.data.summaryIDCounter++;
    const id = store.data.summaryIDCounter;
    const summaryInputTemplate = createSummaryInputTemplate(id);
    const summaryInputDiv = utils.createElementFromTemplate(summaryInputTemplate);
    const child = store.elements.summaryInputContainer.appendChild(summaryInputDiv);
    store.charts.distributions.clear();
    store.charts.distributions.render();
    const summaryInputClear = child.querySelector(`#summary-input-clear-${id}`);
    summaryInputClear.addEventListener("click", () => {
      store.elements.summaryInputContainer.removeChild(child);
      store.data.sessionData = store.data.sessionData.filter((x) => x.id !== `${id}`);
      store.charts.distributions.data = {
        totalRuns: store.data.runCounter,
        sessionData: store.data.sessionData,
        trainingData: store.data.trainingData,
      };
      store.charts.distributions.clear();
      store.charts.distributions.render();
    });
    const summaryInputX = child.querySelector(`#summary-input-${id}`);
    summaryInputX.addEventListener("input", () => summaryInputX.classList.remove("input-error"));
    summaryInputX.addEventListener("keydown", function (e) {
      enableTabbing(e, this);
    });
  });

  // SUMMARY SELECT

  store.elements.summarySelect.addEventListener("change", () => {
    disableControls();
    updateCurrentCharts("clear");

    const summaryID = store.elements.summarySelect.value;
    const summaryOpts = store.data.summaryOptions[summaryID];

    store.data.summarySelectedID = summaryID;

    const perturbation = store.charts.perturbation;
    perturbation.controls.summarySelectedID = summaryID;
    perturbation.controls.render = summaryOpts.perturbation.render;
    perturbation.controls.model = summaryOpts.perturbation.model;
    perturbation.controls.span = summaryOpts.perturbation.span;
    perturbation.controls.colorBy = summaryOpts.perturbation.colorBy;

    const attention = store.charts.attention;
    attention.controls.summarySelectedID = summaryID;
    attention.controls.render = summaryOpts.attention.render;
    attention.controls.model = summaryOpts.attention.model;
    attention.controls.layer = summaryOpts.attention.layer;
    attention.controls.head = summaryOpts.attention.head;
    attention.controls.selectedToken = summaryOpts.attention.selectedToken;

    enableControls(summaryOpts);
    updateCurrentCharts("render");
    if (summaryOpts.attention.render) {
      updateAttentionSliders({
        layer: summaryOpts.attention.layer,
        head: summaryOpts.attention.head,
        token: summaryOpts.attention.selectedToken,
      });
    }
  });

  // DISTRIBUTIONS

  const dxs = store.elements.distributionsXSelect;
  const dys = store.elements.distributionsYSelect;
  const drb = store.elements.distributionsResetButton;

  dxs.addEventListener("change", () => updateChartControl(dxs, "distributions", "xAttr"));
  dys.addEventListener("change", () => updateChartControl(dys, "distributions", "yAttr"));
  drb.addEventListener("click", () => {
    store.charts.distributions.clear();
    store.charts.distributions.render();
  });

  // PERTURBATION

  const pms = store.elements.perturbationModelSelect;
  const pss = store.elements.perturbationSpanSelect;
  const pcbs = store.elements.perturbationColorBySelect;

  pms.addEventListener("change", () => updateChartControl(pms, "perturbation", "model"));
  pss.addEventListener("change", () => updateChartControl(pss, "perturbation", "span"));
  pcbs.addEventListener("change", () => updateChartControl(pcbs, "perturbation", "colorBy"));

  // ATTENTION

  const ams = store.elements.attentionModelSelect;
  const als = store.elements.attentionLayerSlider;
  const ahs = store.elements.attentionHeadSlider;
  const ats = store.elements.attentionTokenSlider;

  ams.addEventListener("change", () => {
    updateAttentionSliders();
    updateChartControl(ams, "attention", "model");
  });
  als.addEventListener("input", () => updateChartControl(als, "attention", "layer"));
  ahs.addEventListener("input", () => updateChartControl(ahs, "attention", "head"));
  ats.addEventListener("input", () => updateChartControl(ats, "attention", "selectedToken"));
}

// HELPERS

function hideWrappers(wrapperIDs) {
  for (const wrapperID of wrapperIDs) {
    const wrapper = document.getElementById(wrapperID);
    const hide = wrapper.getAttribute("hide") !== "true";
    wrapper.setAttribute("hide", hide ? "true" : "false");
  }
}

function enableTabbing(e, self) {
  // Allow tab character to be inserted using tab key
  // See: <https://stackoverflow.com/a/6637396>
  if (e.key == "Tab") {
    e.preventDefault();
    var start = self.selectionStart;
    var end = self.selectionEnd;
    self.value = self.value.substring(0, start) + "\t" + self.value.substring(end);
    self.selectionStart = self.selectionEnd = start + 1;
  }
}

function updateChartControl(selectElem, chartName, control) {
  const value = selectElem.value;

  if (chartName == "perturbation" || chartName == "attention") {
    const summaryID = store.elements.summarySelect.value;
    const summaryOpts = store.data.summaryOptions[summaryID];
    summaryOpts[chartName][control] = value;
  }

  const chart = store.charts[chartName];
  chart.controls[control] = value;
  chart.clear();
  chart.render();
}

function getKeyphrasesHTML(text) {
  return text
    .split(";")
    .filter((x) => x.trim().length > 0)
    .map((x) => `<div>${x.trim()}</div>`)
    .join("");
}

function updateAttentionSliders({ layer = null, head = null, token = null } = {}) {
  const summaryID = parseInt(store.data.summarySelectedID);
  const model = store.elements.attentionModelSelect.value;
  const ntokens = store.data.tokens[summaryID][model].all.length; // number of total tokens
  const las = store.data.attentionsShapes[summaryID][model].local; // local attention shape
  const nlayers = las[0];
  const nheads = las[1];

  document.querySelector(":root").style.setProperty("--nalt", nlayers);
  document.querySelector(":root").style.setProperty("--naht", nheads);

  const layerOpts = [];
  const headOpts = [];
  for (let i = 1; i <= nlayers; i++) {
    const opt = document.createElement("option");
    opt.value = `${i}`;
    opt.label = `${i}`;
    layerOpts.push(opt);
  }
  for (let i = 1; i <= nheads; i++) {
    const opt = document.createElement("option");
    opt.value = `${i}`;
    opt.label = `${i}`;
    headOpts.push(opt);
  }
  store.elements.attentionLayerTicks.replaceChildren(...layerOpts);
  store.elements.attentionHeadTicks.replaceChildren(...headOpts);

  const layerSlider = store.elements.attentionLayerSlider;
  if (layer !== null) {
    layerSlider.value = layer;
  } else if (layerSlider.value >= nlayers) {
    layerSlider.value = nlayers;
  }
  layerSlider.max = nlayers;

  const headSlider = store.elements.attentionHeadSlider;
  if (head !== null) {
    headSlider.value = head;
  } else if (headSlider.value >= nheads) {
    headSlider.value = nheads;
  }
  headSlider.max = nheads;

  const tokenSlider = store.elements.attentionTokenSlider;
  if (token !== null) {
    tokenSlider.value = token;
  } else if (tokenSlider.value >= ntokens) {
    tokenSlider.value = ntokens - 1;
  }
  tokenSlider.max = ntokens - 1;
}

function processRequests(data, requests) {
  const summarySelectOpts = [];
  const perturbationData = {};
  const attentionData = {};

  for (const request of requests) {
    const summaryID = `${request.id}`;

    // get data from streamed chunks
    const inputsRaw = data.shift();
    const scoresRaw = data.shift();
    // const keyphrasesRaw = data.shift();
    const tokensRaw = data.shift();
    const pScoresRaw = data.shift();
    const aShapesRaw = data.shift();

    // format streamed data for visualization components
    const inputs = {
      summary: {
        text: inputsRaw["summary_input"]["text"],
        sentences: inputsRaw["summary_input"]["sentences"],
        words: inputsRaw["summary_input"]["words"],
      },
      source: {
        text: inputsRaw["source_input"]["text"],
        sentences: inputsRaw["source_input"]["sentences"],
        words: inputsRaw["source_input"]["words"],
      },
    };
    // const keyphrases = {
    //   source: keyphrasesRaw["source_keyphrases"],
    // };

    const models = request.models;
    const levels = ["local", "global"];

    const getScores = (model) => scoresRaw[`${model}_score`];
    const getTokens = (model) => {
      return {
        all: tokensRaw[`${model}_tokens`],
        global: tokensRaw[`${model}_global_tokens`],
      };
    };
    const getPScores = (model) => pScoresRaw[`${model}_perturbation_scores`];
    const getAShapes = (model) => {
      return {
        local: aShapesRaw[`${model}_local_attentions_shape`],
        global: aShapesRaw[`${model}_global_attentions_shape`],
      };
    };
    const getAttns = () => {
      return {
        local: [],
        global: [],
      };
    };
    const scores = Object.fromEntries(models.map((model) => [model, getScores(model)]));
    const tokens = Object.fromEntries(models.map((model) => [model, getTokens(model)]));
    const perturbationScores = Object.fromEntries(models.map((model) => [model, getPScores(model)]));
    const attentionsShapes = Object.fromEntries(models.map((model) => [model, getAShapes(model)]));
    const attentions = Object.fromEntries(models.map((model) => [model, getAttns()]));

    // get attention scores
    // local_shape == [layers, heads, seq_len, x + attention_window + 1]
    // global_shape == [layers, heads, seq_len, x]
    if (request.getAttentionScores == "true") {
      let model, level, as, na;
      let startIndex,
        endIndex = 0;

      // MUST BE IN THE SAME ORDER THAT THE SERVER SENDS THE LISTS
      const models_levels = models.flatMap((m) => levels.map((l) => [m, l]));
      models_levels.forEach((model_level) => {
        model = model_level[0];
        level = model_level[1];
        as = attentionsShapes[model][level]; // attention scores
        na = as[0] * as[1] * as[2]; // number of arrays to subset
        startIndex = endIndex;
        endIndex = startIndex + na;
        attentions[model][level] = data.slice(startIndex, endIndex);
      });
      data.splice(0, endIndex); // remove data to prepare for next request

      store.data.tokens[summaryID] = tokens;
      store.data.attentionsShapes[summaryID] = attentionsShapes;
    }

    // save scores
    const tableRow = {
      id: summaryID,
      run: request.run,
    };
    Object.keys(store.models).forEach((x) => (tableRow[x] = models.includes(x) ? scores[x] : "-"));
    store.data.scoresTableData.push(tableRow);
    store.tables.scoresTable.replaceData(store.data.scoresTableData);

    // save keyphrases
    // const sourceKeyphrasesHTML = getKeyphrasesHTML(keyphrases.source);
    // store.elements.sourceKeyphrases.innerHTML = sourceKeyphrasesHTML;

    // save chart data
    store.data.sessionData.push({
      id: summaryID,
      run: request.run,
      summary: inputs.summary.text,
      source: inputs.source.text,
      scores: scores,
      // sourceKeyphrases: keyphrases.source,
    });
    perturbationData[summaryID] = { scores, perturbationScores };
    attentionData[summaryID] = { tokens, attentionsShapes, attentions };

    // save options
    const enableGrammarPerturbation = request.getGrammarPerturbation == "true";
    const enableSentencePerturbation = request.getSentencePerturbation == "true";
    const enableWordPerturbation = request.getWordPerturbation == "true";
    const enableTokenPerturbation = request.getTokenPerturbation == "true";
    const renderPerturbation =
      enableGrammarPerturbation || enableSentencePerturbation || enableWordPerturbation || enableTokenPerturbation;
    const perturbationSpan = enableGrammarPerturbation
      ? "grammar"
      : enableSentencePerturbation
      ? "sentence"
      : enableWordPerturbation
      ? "word"
      : enableTokenPerturbation
      ? "token"
      : "";
    const perturbationModel = models[0];
    const perturbationColorBy = "diff_true";

    const renderAttention = request.getAttentionScores == "true";
    const attentionModel = models[0];

    const opts = {
      models,
      perturbation: {
        render: renderPerturbation,
        enable: {
          grammar: enableGrammarPerturbation,
          sentence: enableSentencePerturbation,
          word: enableWordPerturbation,
          token: enableTokenPerturbation,
        },
        span: perturbationSpan,
        model: perturbationModel,
        colorBy: perturbationColorBy,
      },
      attention: {
        render: renderAttention,
        model: attentionModel,
        layer: "1",
        head: "1",
        selectedToken: "0",
      },
    };
    store.data.summaryOptions[summaryID] = opts;

    // create summary select option
    const opt = document.createElement("option");
    opt.value = summaryID;
    opt.innerHTML = summaryID;
    summarySelectOpts.push(opt);
  }

  store.elements.summarySelect.replaceChildren(...summarySelectOpts);

  return [perturbationData, attentionData];
}

function createSummaryInputTemplate(id) {
  return `
    <div id="summary-input-wrapper-${id}" class="summary-input-wrapper flex-row cg-4" data-id="${id}">
      <textarea
        id="summary-input-${id}"
        class="summary-input frame"
        placeholder="Type or copy/paste a summary..."
      ></textarea>
      <div class="flex-col rg-4">
        <div class="summary-input-options flex-row">
          <b>ID:&nbsp;${id}</b>
          <a id="summary-input-clear-${id}" class="summary-input-clear"></a>
        </div>
        <div class="flex-col rg-4">
          <b>✏️ Perturb:</b>
          <div class="flex-row">
            <input
              type="checkbox"
              id="grammar-perturbation-checkbox-${id}"
              name="grammar-perturbation-checkbox-${id}"
            />
            <label for="grammar-perturbation-checkbox-${id}">grammar?</label>
          </div>
          <div class="flex-row">
            <input
              type="checkbox"
              id="sentence-perturbation-checkbox-${id}"
              name="sentence-perturbation-checkbox-${id}"
            />
            <label for="sentence-perturbation-checkbox-${id}">sentences?</label>
          </div>
          <div class="flex-row">
            <input
              type="checkbox"
              id="word-perturbation-checkbox-${id}"
              name="word-perturbation-checkbox-${id}"
            />
            <label for="word-perturbation-checkbox-${id}">words?</label>
          </div>
          <div class="flex-row">
            <input
              type="checkbox"
              id="token-perturbation-checkbox-${id}"
              name="token-perturbation-checkbox-${id}"
            />
            <label for="token-perturbation-checkbox-${id}">tokens?</label>
          </div>
        </div>
        <div class="flex-col rg-4">
          <b>👀 Attention:</b>
          <div class="flex-row">
            <input type="checkbox" id="get-attention-checkbox-${id}" name="get-attention-checkbox-${id}" />
            <label for="get-attention-checkbox-${id}">tokens?</label>
          </div>
        </div>
      </div>
    </div>`;
}
