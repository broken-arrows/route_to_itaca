/* =============================================================================
 * source/lib/allegiances.js — GAME model, handed to the engine via the lib.
 * =============================================================================
 * Which parties a person has belonged to, as a function of live game state
 * (Q). This is irreducible Q-branching logic — "show DL only once Q.dl_formed",
 * and a few entries branch into DIFFERENT label sets — so it is CODE, not a
 * lookup table, and cannot live in glossary.json (which is data, and must stay
 * data so its text can be translated). It is keyed by the SAME glossary term id
 * the registry assigns, so a UI composes the two at render time:
 * glossary[id] (static facts) ⊕ allegiances[id](Q) (dynamic facts).
 *
 * Each function returns an ARRAY of PRESENTATION-NEUTRAL entries:
 *     { colour, label, note? }
 * `colour` is a token name ("erc") or, where the old palette never had a var
 * for a party, a raw hex ("#555555") — the same token-or-hex convention as
 * glossary.json. `note` (e.g. "former") is optional trailing plain text. NO
 * markup, NO CSS: each UI maps the token to its own palette and renders its own
 * span. (Ported from out/html/data.js's tooltipList closures via game.js.)
 *
 * Dual-consumable like cat_engine.js: Vite/vitest `require` it; the old
 * bundler-less shell loads it as a plain <script>, so it also publishes
 * window.RTI_ALLEGIANCES. Pure and DOM-free — reads only Q.
 * ========================================================================== */
(function () {
  "use strict";

  var allegiances = {
    llu_s_companys: function (Q) {
      return [{ colour: "erc", label: "ERC" }];
    },
    francisco_franco: function (Q) {
      return [{ colour: "#555555", label: "FET y de las JONS" }];
    },
    jordi_pujol: function (Q) {
      return [
        { colour: "ciu", label: "CDC" },
        { colour: "ciu", label: "CiU" },
      ];
    },
    artur_mas: function (Q) {
      var list = [
        { colour: "cdc", label: "CDC" },
        { colour: "ciu", label: "CiU" },
      ];
      if (Q.dl_formed) list.push({ colour: "dl", label: "DL" });
      if (Q.jxsi_formed) list.push({ colour: "jxsi", label: "JxSí" });
      if (Q.pdcat_formed) list.push({ colour: "pdcat", label: "PDeCAT" });
      if (Q.jxcat_formed && !Q.mas_ousted)
        list.push({ colour: "jxcat", label: "JxCat" });
      if (Q.junts_formed && !Q.mas_ousted)
        list.push({ colour: "junts", label: "Junts" });
      return list;
    },
    carles_puigdemont: function (Q) {
      var list = [
        { colour: "cdc", label: "CDC" },
        { colour: "ciu", label: "CiU" },
      ];
      if (Q.dl_formed) list.push({ colour: "dl", label: "DL" });
      if (Q.jxsi_formed) list.push({ colour: "jxsi", label: "JxSí" });
      if (Q.pdcat_formed && !Q.pdcat_split) {
        list.push({ colour: "pdcat", label: "PDeCAT" });
      } else if (Q.pdcat_formed) {
        list.push({ colour: "pdcat", label: "PDeCAT", note: "former" });
      }
      if (Q.jxcat_formed) list.push({ colour: "jxcat", label: "JxCat" });
      if (Q.junts_formed) list.push({ colour: "junts", label: "Junts" });
      return list;
    },
    josep_antoni_duran_i_lleida: function (Q) {
      return [
        { colour: "unio", label: "UDC" },
        { colour: "ciu", label: "CiU" },
      ];
    },
    oriol_junqueras: function (Q) {
      var list = [{ colour: "erc", label: "ERC" }];
      if (Q.jxsi_formed) list.push({ colour: "jxsi", label: "JxSí" });
      if (Q.jxcat_formed && Q.erc_in_jxcat)
        list.push({ colour: "jxcat", label: "JxCat" });
      return list;
    },
    david_fern_ndez: function (Q) {
      return [{ colour: "#b8a12b", label: "CUP" }];
    },
    benet_salellas: function (Q) {
      return [{ colour: "#b8a12b", label: "CUP" }];
    },
    dolors_sabater: function (Q) {
      return [{ colour: "#b8a12b", label: "CUP" }];
    },
    david_fern_ndez: function (Q) {
      return [{ colour: "#b8a12b", label: "CUP" }];
    },
    carles_riera: function (Q) {
      return [
        { colour: "#c50000", label: "Endavant" },
        { colour: "#b8a12b", label: "CUP" },
      ];
    },
    mireia_vehi: function (Q) {
      return [
        { colour: "#c50000", label: "Endavant" },
        { colour: "#b8a12b", label: "CUP" },
      ];
    },
    vial_aragones: function (Q) {
      return [
        { colour: "#c50000", label: "Endavant" },
        { colour: "#b8a12b", label: "CUP" },
      ];
    },
    eulalia_reguant: function (Q) {
      return [
        { colour: "#c50000", label: "Endavant" },
        { colour: "#b8a12b", label: "CUP" },
      ];
    },
    anna_gabriel: function (Q) {
      return [
        { colour: "#c50000", label: "Endavant" },
        { colour: "#b8a12b", label: "CUP" },
      ];
    },
    albert_botran: function (Q) {
      return [
        { colour: "#c11d4e", label: "Poble Lliure" },
        { colour: "#b8a12b", label: "CUP" },
      ];
    },
    antonio_banos: function (Q) {
      if (Q.jxsi_formed && Q.pressing_cup_happened) {
        return [{ colour: "indp", label: "indp." }];
      }
      return [
        { colour: "#c11d4e", label: "Poble Lliure" },
        { colour: "#b8a12b", label: "CUP" },
      ];
    },
    mireia_boya: function (Q) {
      return [
        { colour: "#c11d4e", label: "Poble Lliure" },
        { colour: "#b8a12b", label: "CUP" },
      ];
    },
    albert_rivera: function (Q) {
      return [{ colour: "cs", label: "Cs" }];
    },
    alfons_l_pez_tena: function (Q) {
      return [{ colour: "si", label: "SI" }];
    },
    pere_navarro: function (Q) {
      var list = [{ colour: "psc", label: "PSC" }];
      if (Q.psc_leader != "Pere Navarro")
        list.push({ colour: "psoe", label: "PSOE" });
      return list;
    },
    al_cia_s_nchez_camacho: function (Q) {
      return [
        { colour: "pp", label: "PP" },
        { colour: "ppc", label: "PPC" },
      ];
    },
    joan_herrera: function (Q) {
      var list = [{ colour: "icv", label: "ICV-EUiA" }];
      if (Q.csqp_formed) list.push({ colour: "csqp", label: "CSQP" });
      return list;
    },
    arcadi_oliveres: function (Q) {
      var list = [{ colour: "#222222", label: "Procés Constituent" }];
      if (Q.csqp_formed && Q.csqp_leader == "Arcadi Oliveres") {
        list.push({ colour: "csqp", label: "CSQP" });
      }
      if (Q.cecp_formed && Q.cecp_leader == "Arcadi Oliveres") {
        list.push({ colour: "cecp", label: "CECP" });
      }
      return list;
    },
    lluis_rabell: function (Q) {
      if (Q.csqp_formed) {
        return [{ colour: "csqp", label: "CSQP" }];
      } else {
        return [{ colour: "indp", label: "indp." }];
      }
    },
    mariano_rajoy: function (Q) {
      return [{ colour: "pp", label: "PP" }];
    },
    pedro_s_nchez: function (Q) {
      return [{ colour: "psoe", label: "PSOE" }];
    },
    eduardo_madina: function (Q) {
      return [
        { colour: "psc", label: "PSE-EE" },
        { colour: "psoe", label: "PSOE" },
      ];
    },
    susana_d_az: function (Q) {
      return [
        { colour: "psoe", label: "PSOE-A" },
        { colour: "psoe", label: "PSOE" },
      ];
    },
    felipe_gonz_lez: function (Q) {
      return [{ colour: "psoe", label: "PSOE" }];
    },
    alfredo_p_rez_rubalcaba: function (Q) {
      return [{ colour: "psoe", label: "PSOE" }];
    },
    joan_laporta_i_estruch: function (Q) {
      return [
        { colour: "si", label: "SI" },
        { colour: "erc", label: "ERC" },
      ];
    },
    ada_colau: function (Q) {
      var list = [{ colour: "bcomu", label: "BComú" }];
      if (Q.csqp_formed) list.push({ colour: "csqp", label: "CSQP" });
      if (Q.cecp_formed) list.push({ colour: "cecp", label: "CECP" });
      if (Q.ecp_formed) list.push({ colour: "ecp", label: "ECP" });
      return list;
    },
    _ngel_ros: function (Q) {
      return [{ colour: "psc", label: "PSC" }];
    },
    montserrat_tura: function (Q) {
      var list = [];
      if (
        !(Q.psc_implosion_countdown < 0) ||
        Q.psc_leader == "Montserrat Tura" ||
        Q.psc_leader == "Àngel Ros"
      ) {
        list.push({ colour: "psc", label: "PSC" });
      } else {
        list.push({ colour: "psc", label: "PSC", note: "former" });
        list.push({ colour: "indp", label: "indp." });
      }
      return list;
    },
    ernest_maragall: function (Q) {
      var list = [];
      if (Q.psc_leader == "Montserrat Tura") {
        list.push({ colour: "psc", label: "PSC" });
      } else {
        list.push({ colour: "psc", label: "PSC", note: "former" });
      }
      if (Q.ernest_maragall_advisor_available) {
        list.push({ colour: "erc", label: "ERC" });
      } else {
        list.push({ colour: "indp", label: "indp." });
      }
      return list;
    },
    pasqual_maragall: function (Q) {
      return [{ colour: "psc", label: "PSC" }];
    },
    n_ria_parlon: function (Q) {
      if (Q.art155_ever) {
        return [
          { colour: "psc", label: "PSC", note: "former" },
          { colour: "indp", label: "indp." },
        ];
      } else {
        return [{ colour: "psc", label: "PSC" }];
      }
    },
    miquel_iceta: function (Q) {
      return [{ colour: "psc", label: "PSC" }];
    },
    gabriel_rufian: function (Q) {
      return [{ colour: "erc", label: "ERC" }];
    },
    soraya_s_enz_de_santamar_a: function (Q) {
      return [{ colour: "pp", label: "PP" }];
    },
    ra_l_romeva: function (Q) {
      if (!Q.jxsi_formed) {
        return [{ colour: "icv", label: "ICV-EUiA" }];
      } else {
        if (Q.jxcat_formed && Q.erc_in_jxcat) {
          return [
            { colour: "icv", label: "ICV-EUiA", note: "former" },
            { colour: "jxsi", label: "JxSí" },
            { colour: "jxcat", label: "JxCat" },
            { colour: "erc", label: "ERC" },
          ];
        } else if (Q.jxcat_formed && !Q.erc_in_jxcat) {
          return [
            { colour: "icv", label: "ICV-EUiA", note: "former" },
            { colour: "jxsi", label: "JxSí" },
            { colour: "erc", label: "ERC" },
          ];
        } else {
          return [
            { colour: "icv", label: "ICV-EUiA", note: "former" },
            { colour: "jxsi", label: "JxSí" },
          ];
        }
      }
    },
    pablo_iglesias: function (Q) {
      if (Q.iu_in_up) {
        return [
          { colour: "podemos", label: "Podemos" },
          { colour: "up", label: "UP" },
        ];
      } else {
        return [{ colour: "podemos", label: "Podemos" }];
      }
    },
    jaume_asens: function (Q) {
      var list = [{ colour: "bcomu", label: "BComú" }];
      if (Q.csqp_formed) list.push({ colour: "csqp", label: "CSQP" });
      if (Q.cecp_formed) list.push({ colour: "cecp", label: "CECP" });
      if (Q.ecp_formed) list.push({ colour: "ecp", label: "ECP" });
      return list;
    },
    joan_coscubiela: function (Q) {
      var list = [{ colour: "icv", label: "ICV-EUiA" }];
      if (Q.csqp_formed) list.push({ colour: "csqp", label: "CSQP" });
      if (Q.cecp_formed) list.push({ colour: "cecp", label: "CECP" });
      if (Q.ecp_formed) list.push({ colour: "ecp", label: "ECP" });
      return list;
    },
    xavier_domenech: function (Q) {
      var list = [{ colour: "#222222", label: "Procés Constituent" }];
      if (Q.csqp_formed) list.push({ colour: "csqp", label: "CSQP" });
      if (Q.cecp_formed) list.push({ colour: "cecp", label: "CECP" });
      if (Q.ecp_formed) list.push({ colour: "ecp", label: "ECP" });
      return list;
    },
    albano_dantefachin: function (Q) {
      var list;
      if (!Q.fr_formed) {
        list = [{ colour: "podemos", label: "Podemos" }];
        if (Q.csqp_formed) list.push({ colour: "csqp", label: "CSQP" });
        if (Q.cecp_formed) list.push({ colour: "cecp", label: "CECP" });
      } else {
        list = [
          { colour: "podemos", label: "Podemos", note: "former" },
          { colour: "csqp", label: "CSQP", note: "former" },
        ];
        if (Q.spa_fr_active) {
          list.push({ colour: "fr", label: "FR" });
        } else {
          list.push({ colour: "fr", label: "FR", note: "former" });
          list.push({ colour: "indp", label: "indp." });
        }
      }
      return list;
    },
    elisenda_alamany: function (Q) {
      if (!Q.cecp_formed) {
        return [{ colour: "indp", label: "indp." }];
      }
      if (Q.ecp_formed) {
        var list = [{ colour: "cecp", label: "CECP", note: "former" }];
        // "From May 2019 onward" — NOT `year >= 2019 && month >= 5`, which is
        // false for Jan–Apr of every later year (masked today only because
        // reaching 2020 ends the game).
        if (Q.year > 2019 || (Q.year == 2019 && Q.month >= 5)) {
          list.push({ colour: "erc", label: "ERC" });
        } else {
          list.push({ colour: "indp", label: "indp." });
        }
        return list;
      }
      return [{ colour: "cecp", label: "CECP" }];
    },
  };

  var api = { allegiances: allegiances };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api; // Vite / vitest
  } else {
    window.RTI_ALLEGIANCES = api; // the old shell: no bundler, script tag only
  }
})();
