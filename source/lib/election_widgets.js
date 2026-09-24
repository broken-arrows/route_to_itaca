/* Pure view models for the election maps shared by both game interfaces. */
(function () {
  "use strict";

  var CITY_IDS = {
    Barcelona: "barcelona",
    Lleida: "lleida",
    Tarragona: "tarragona",
    Reus: "reus",
    Girona: "girona",
    "Vielha e Mijaran": "vielha-e-mijaran",
    "Vilafranca del Penedès": "vilafranca-del-penedes",
    "L'Hospitalet de Llobregat": "lhospitalet-de-llobregat",
    "Sant Boi de Llobregat": "sant-boi-de-llobregat",
    "El Prat de Llobregat": "el-prat-de-llobregat",
    Viladecans: "viladecans",
    "Esplugues de Llobregat": "esplugues-de-llobregat",
    Ripollet: "ripollet",
    "Sant Adrià de Besòs": "sant-adria-de-besos",
    Terrassa: "terrassa",
    Sabadell: "sabadell",
    "Cornellà de Llobregat": "cornella-de-llobregat",
    Rubí: "rubi",
    Granollers: "granollers",
    "Mollet del Vallès": "mollet-del-valles",
    Mataró: "mataro",
    Badalona: "badalona",
    "Santa Coloma de Gramenet": "santacolomadegramenet",
    Balaguer: "balaguer",
    "Sant Vicenç dels Horts": "sant-vicenc-dels-horts",
    Martorell: "martorell",
    "Vilafranca del Penedès": "vilafranca-del-penedes",
    "Sant Cugat del Vallès": "sant-cugat-del-valles",
    Manresa: "manresa",
    Figueres: "figueres",
    Vic: "vic",
    Igualada: "igualada",
    Olot: "olot",
    "La Seu d'Urgell": "la-seu-durgell",
    Tàrrega: "tarrega",
    Berga: "berga",
    Solsona: "solsona",
    Ripoll: "ripoll",
    Tortosa: "tortosa",
    Amposta: "amposta",
    Banyoles: "banyoles",
    "Vilanova i la Geltrú": "vilanova-i-la-geltru",
    Tremp: "tremp",
    "Móra d'Ebre": "mora-debre",
  };

  var CONGRESO_REGIONS = {
    catalonia: ["catalunya", "Catalonia"],
    valencia: ["valencia", "Valencia"],
    balearic_islands: ["balears", "Balearic Islands"],
    navarre: ["navarra", "Navarre"],
    basque_country: ["euskadi", "Basque Country"],
    galicia: ["galicia", "Galicia"],
  };
  var REST_PROVINCES = [
    "madrid", "andalusia", "extremadura", "castile_la_mancha",
    "castile_and_leon", "murcia", "la_rioja", "aragon", "asturias",
    "cantabria", "canary_islands", "ceuta", "melilla",
  ];

  function congresoResultsMap(Q) {
    Q = Q || {};
    var regions = Object.keys(CONGRESO_REGIONS).map(function (svgId) {
      var constituency = CONGRESO_REGIONS[svgId][0];
      return {
        id: svgId,
        constituency: constituency,
        winner: Q["congreso_" + constituency + "_wp"] || null,
      };
    });
    REST_PROVINCES.forEach(function (id) {
      regions.push({
        id: id,
        constituency: "rest",
        winner: Q["congreso_prov_" + id + "_wp"] || null,
      });
    });
    var names = { rest: "Rest of Spain" };
    Object.keys(CONGRESO_REGIONS).forEach(function (svgId) {
      names[CONGRESO_REGIONS[svgId][0]] = CONGRESO_REGIONS[svgId][1];
    });
    var panels = {};
    Object.keys(names).forEach(function (constituency) {
      var parties = Q["congreso_parties_" + constituency] || [];
      var rows = (Array.isArray(parties) ? parties : []).map(function (party) {
        return {
          party: party,
          seats: Number(Q[party + "_congreso_s_" + constituency]) || 0,
          support: Number(Q[party + "_congreso_" + constituency + "_support"]) || 0,
        };
      }).filter(function (row) { return row.seats > 0; });
      rows.sort(function (a, b) {
        return b.support - a.support || b.seats - a.seats;
      });
      panels[constituency] = {
        name: names[constituency],
        seats: Number(Q.congreso_seats && Q.congreso_seats[constituency]) || 0,
        rows: rows,
      };
    });
    return { regions: regions, panels: panels };
  }

  function localResultsMap(Q) {
    Q = Q || {};
    var winners = {};
    Object.keys(Q).forEach(function (key) {
      var match = /^local_(.+)_wp$/.exec(key);
      if (match && Q[key]) winners[match[1]] = Q[key];
    });
    return {
      winners: winners,
      cities: Object.keys(CITY_IDS).map(function (name) {
        return { name: name, id: CITY_IDS[name], winner: winners[CITY_IDS[name]] || null };
      }),
    };
  }

  function congresoPartyTour(Q) {
    Q = Q || {};
    var leftParty = Q.iu_in_up === true ? "up" : "podemos";
    var spa = ["psoe", "pp", leftParty, "csspa"];
    if (Q.spa_cs_active) spa.push("csspa");
    if (!Q.iu_in_up) spa.push("iu");
    if (Q.spa_mpais_active) spa.push("mpais");
    if (Q.vox_active) spa.push("vox");
    if (Q.upyd_congreso_s > 0) spa.push("upyd");

    var catalonia = [];
    if (Q.junts_formed) catalonia.push("junts");
    else if (Q.jxcat_formed) catalonia.push("jxcat");
    else if (Q.pdcat_formed && !Q.pdcat_split) catalonia.push("pdcat");
    else if (Q.dl_formed) catalonia.push("dl");
    else if (Q.unio_split) catalonia.push("cdc");
    else if (!Q.jxsi_in_congreso) catalonia.push("ciu");
    if (Q.jxsi_in_congreso) catalonia.push("jxsi");
    if (!Q.erc_in_jxcat && !Q.jxsi_in_congreso) catalonia.push("erc");
    if (Q.psc_split) catalonia.push("psc");
    if (Q.spa_cup_active) catalonia.push("cup");
    if (Q.spa_fr_active) catalonia.push("fr");
    if (Q.pdcat_split && Q.pdcat_congreso_catalunya_support > 0) catalonia.push("pdcat");
    if (Q.unio_split && Q.unio_congreso_catalunya_support > 0) catalonia.push("unio");

    var ppcc = [];
    if (Q.spa_compromis_active || Q.party_reaching_out_map) ppcc.push("compromis");
    if (Q.spa_mes_active || Q.party_reaching_out_map) ppcc.push("mesm");
    var eh = ["pnv"];
    if (!Q.party_reaching_out_map && !Q.spa_nsuma_formed && Q.upn_in_pp === false) eh.push("upn");
    if (!Q.party_reaching_out_map && Q.spa_nsuma_formed) eh.push("nsuma");
    if (Q.spa_ehbildu_active) eh.push("ehbildu", "gbai");
    else eh.push("amaiur");
    var galicia = [];
    if (Q.spa_bng_active || Q.party_reaching_out_map) galicia.push("bng");
    var others = [];
    if (Q.spa_foro_active) others.push("fac");
    if (Q.spa_te_active) others.push("texiste");
    others.push("prc", "cc");

    var highlighted = Array.isArray(Q.congreso_party_tour_highlight)
      ? Q.congreso_party_tour_highlight : [];
    var parties = { rest: spa, catalonia: catalonia, ppcc: ppcc, eh: eh, galicia: galicia, others: others };
    return Object.keys(parties).map(function (region) {
      return {
        id: region,
        viewed: !!Q["congreso_party_tour_viewed_" + region],
        highlighted: region !== "rest" && highlighted.includes(region),
        revealedByRest: highlighted.includes("rest"),
        parties: Array.from(new Set(parties[region])),
        aragon: !!Q.spa_te_active,
      };
    });
  }

  var api = {
    congresoResultsMap: congresoResultsMap,
    localResultsMap: localResultsMap,
    congresoPartyTour: congresoPartyTour,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else window.RTI_ELECTION_WIDGETS = api;
})();
