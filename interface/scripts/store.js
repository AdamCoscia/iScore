export default class store {
  constructor({ utils, Examples, TrainingData, Distributions, Perturbation, Attention }) {
    this.backendURL = {
      connect: "http://localhost:8001/connect",
      getData: "http://localhost:8001/get-data",
    };
    this.models = {
      // MUST BE IN THE SAME ORDER THAT THE SERVER SENDS THEM
      slc: {
        name: "summary-longformer-content",
        label: "Content",
      },
      lcg: {
        name: "longformer-content-global",
        label: "Content (global)",
      },
      slw: {
        name: "summary-longformer-wording",
        label: "Wording",
      },
      lwg: {
        name: "longformer-wording-global",
        label: "Wording (global)",
      },
    };
    this.charts = {
      distributions: new Distributions(utils, d3.select("#distributions")),
      perturbation: new Perturbation(utils, d3.select("#perturbation")),
      attention: new Attention(utils, d3.select("#attention")),
    };
    this.tables = {
      scoresTable: null,
    };
    this.data = {
      examples: Examples,
      trainingData: TrainingData,
      sessionData: [],
      scoresTableData: [],
      tokens: {},
      attentionsShapes: {},
      summaryOptions: {},
      summaryIDCounter: 1,
      summarySelectedID: "1",
      runCounter: 1,
    };
    this.elements = {
      dataQueryButton: document.getElementById("data-query-button"),
      modelCheckboxesHide: document.getElementById("model-checkboxes-hide"),
      modelCheckboxes: document.getElementById("model-checkboxes-container"),
      examplesSelect: document.getElementById("examples-select"),

      sourceFileUpload: document.getElementById("source-file-upload"),
      sourceUploadButton: document.getElementById("source-upload-button"),
      sourceInput: document.getElementById("source-input"),
      // sourceKeyphrases: document.getElementById("source-keyphrases"),

      summaryFileUpload: document.getElementById("summary-file-upload"),
      summaryUploadButton: document.getElementById("summary-upload-button"),
      summaryInputContainer: document.getElementById("summary-input-container"),
      summaryInput1: document.getElementById("summary-input-1"),
      summaryAddButton: document.getElementById("summary-add-button"),

      showScoresCheckbox: document.getElementById("show-scores-checkbox"),
      showDistributionsCheckbox: document.getElementById("show-distributions-checkbox"),

      distributionsXSelect: document.getElementById("distributions-x-select"),
      distributionsYSelect: document.getElementById("distributions-y-select"),
      distributionsResetButton: document.getElementById("distributions-reset-button"),

      showPerturbationCheckbox: document.getElementById("show-perturbation-checkbox"),
      showAttentionCheckbox: document.getElementById("show-attention-checkbox"),

      scores: document.getElementById("scores"),
      scoresTable: document.getElementById("scores-table"),

      summarySelect: document.getElementById("summary-select"),

      perturbationModelSelect: document.getElementById("perturbation-model-select"),
      perturbationSpanSelect: document.getElementById("perturbation-span-select"),
      perturbationColorBySelect: document.getElementById("perturbation-colorby-select"),

      attentionModelSelect: document.getElementById("attention-model-select"),
      attentionHeatmapsCheckbox: document.getElementById("attention-heatmaps-checkbox"),
      attentionLayerSlider: document.getElementById("attention-layer-slider"),
      attentionLayerTicks: document.getElementById("attention-layer-ticks"),
      attentionHeadSlider: document.getElementById("attention-head-slider"),
      attentionHeadTicks: document.getElementById("attention-head-ticks"),
      attentionTokenSlider: document.getElementById("attention-token-slider"),
    };
  }
}
