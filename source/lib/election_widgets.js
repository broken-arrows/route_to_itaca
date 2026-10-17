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
    "madrid",
    "andalusia",
    "extremadura",
    "castile_la_mancha",
    "castile_and_leon",
    "murcia",
    "la_rioja",
    "aragon",
    "asturias",
    "cantabria",
    "canary_islands",
    "ceuta",
    "melilla",
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
      var rows = (Array.isArray(parties) ? parties : [])
        .map(function (party) {
          return {
            party: party,
            seats: Number(Q[party + "_congreso_s_" + constituency]) || 0,
            support:
              Number(Q[party + "_congreso_" + constituency + "_support"]) || 0,
          };
        })
        .filter(function (row) {
          return row.seats > 0;
        });
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
        return {
          name: name,
          id: CITY_IDS[name],
          winner: winners[CITY_IDS[name]] || null,
        };
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
    if (Q.pdcat_split && Q.pdcat_congreso_catalunya_support > 0)
      catalonia.push("pdcat");
    if (Q.unio_split && Q.unio_congreso_catalunya_support > 0)
      catalonia.push("unio");

    var ppcc = [];
    if (Q.spa_compromis_active || Q.party_reaching_out_map)
      ppcc.push("compromis");
    if (Q.spa_mes_active || Q.party_reaching_out_map) ppcc.push("mesm");
    var eh = ["pnv"];
    if (
      !Q.party_reaching_out_map &&
      !Q.spa_nsuma_formed &&
      Q.upn_in_pp === false
    )
      eh.push("upn");
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
      ? Q.congreso_party_tour_highlight
      : [];
    var parties = {
      rest: spa,
      catalonia: catalonia,
      ppcc: ppcc,
      eh: eh,
      galicia: galicia,
      others: others,
    };
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

  // The builder edits a draft. Published seat qualities and both charts remain
  // untouched until the authored Run choice commits it.
  function coalitionBuilderConfig(Q, chamber, parties) {
    if (chamber === "parlament") return parlamentBuilderConfig(Q, parties);
    if (chamber === "congreso") return congresoBuilderConfig(Q, parties);
    var constituencies = Q[chamber + "_constituencies"] || [];
    return {
      chamber: chamber,
      rows: parties
        .filter(function (party) {
          var id = party[0];
          if (
            Q[id + "_" + chamber + "_s"] > 0 ||
            Q[id + "_" + chamber + "_pv"] > 0
          )
            return true;
          return constituencies.some(function (c) {
            if (chamber === "congreso")
              return Q[id + "_congreso_" + c + "_support"] > 0;
            return (Q.parlament_demographics || []).some(function (d) {
              return Q[id + "_parlament_" + c + "_" + d + "_support"] > 0;
            });
          });
        })
        .map(function (party) {
          return {
            id: party[0],
            name: party[1],
            seats: Number(Q[party[0] + "_" + chamber + "_s"]) || 0,
          };
        }),
    };
  }

  function builderPartyRow(
    Q,
    chamber,
    names,
    slot,
    variants,
    column,
    position,
  ) {
    var id =
      variants.find(function (p) {
        return Q[p + "_" + chamber + "_s"] > 0;
      }) || variants[0];
    return {
      slot: slot,
      id: id,
      name: names[id],
      variants: variants,
      column: column,
      position: position,
      seats: variants.reduce(function (sum, p) {
        return sum + (Number(Q[p + "_" + chamber + "_s"]) || 0);
      }, 0),
    };
  }

  function congresoBuilderConfig(Q, parties) {
    var names = Object.fromEntries(parties);
    // Sumar is outside this editor's scope; retain its live seats on Run.
    delete names.sumar;
    function row(slot, variants, column, position) {
      return builderPartyRow(
        Q,
        "congreso",
        names,
        slot,
        variants,
        column,
        position,
      );
    }
    var up = row("up", ["up"], 0, 1);
    up.isJoint = true;
    var joint = row("joint", ["jxsi", "jxcat"], 0, 6);
    joint.isJoint = true;
    var core = row("ciu", ["ciu", "cdc", "dl", "pdcat", "junts"], 1, 6);
    var mainBallot = core.variants.find(function (p) {
      return p !== "pdcat" && Q[p + "_congreso_s"] > 0;
    });
    if (mainBallot) {
      core.id = mainBallot;
      core.name = names[mainBallot];
    }
    var separate = row("pdcat", ["pdcat"], 2, 9);
    if (core.id !== "pdcat") core.seats -= separate.seats;
    else separate.seats = 0;
    var rows = [
      up,
      joint,
      row("iu", ["iu"], 1, 1),
      row("podemos", ["podemos"], 1, 2),
      row("compromis", ["compromis"], 1, 3),
      row("mes", ["mes"], 1, 4),
      row("bng", ["bng"], 1, 5),
      core,
      row("erc", ["erc"], 1, 7),
      row("cup", ["cup"], 1, 8),
      row("fr", ["fr"], 1, 9),
      row("pnv", ["pnv"], 2, 1),
      row("amaiur", ["amaiur", "ehbildu"], 2, 2),
      row("gbai", ["gbai"], 2, 3),
      row("cc", ["cc"], 2, 4),
      row("mpais", ["mpais"], 2, 5),
      row("prc", ["prc"], 2, 6),
      row("te", ["te"], 2, 7),
      row("unio", ["unio"], 2, 8),
      separate,
      row("psoe", ["psoe"], 3, 1),
      row("psc", ["psc"], 3, 2),
      row("pp", ["pp"], 3, 3),
      row("cs", ["cs"], 3, 4),
      row("upn", ["upn", "nsuma"], 3, 5),
      row("fac", ["fac"], 3, 6),
      row("upyd", ["upyd"], 3, 7),
      row("vox", ["vox"], 3, 8),
    ];
    var config = { chamber: "congreso", names: names, rows: rows };
    rows
      .filter(function (entry) {
        return entry.column === 1;
      })
      .forEach(function (entry) {
        if (entry.slot === "fr") {
          entry.group = null;
          entry.canJoin = false;
          entry.member = false;
          entry.contribution = 0;
          return;
        }
        entry.group = entry.position <= 5 ? "up" : "joint";
        entry.canJoin = true;
        entry.member = false;
        entry.contribution = 0;
        var list = entry.group === "up" ? up : joint;
        var selected =
          entry.group === "up"
            ? !entry.canJoin || Q["spa_" + entry.slot + "_active"] === false
            : !entry.canJoin || !!Q[entry.id + "_in_" + joint.id];
        if (list.seats > 0 && selected) setBuilderMember(config, entry, true);
      });
    up.disabled = false;
    joint.disabled = false;
    syncCoalitionBuilder(config);
    return config;
  }

  function parlamentBuilderConfig(Q, parties) {
    var names = Object.fromEntries(parties);
    function row(slot, variants, column, position) {
      var id =
        variants.find(function (p) {
          return Q[p + "_parlament_s"] > 0;
        }) || variants[0];
      return {
        slot: slot,
        id: id,
        name: names[id],
        variants: variants,
        column: column,
        position: position,
        seats: variants.reduce(function (sum, p) {
          return sum + (Number(Q[p + "_parlament_s"]) || 0);
        }, 0),
      };
    }
    var core = row("ciu", ["ciu", "cdc", "dl", "pdcat", "junts"], 1, 1);
    var mainBallot = core.variants.find(function (p) {
      return p !== "pdcat" && Q[p + "_parlament_s"] > 0;
    });
    if (mainBallot) {
      core.id = mainBallot;
      core.name = names[mainBallot];
    }
    // A split PDeCAT keeps its own ballot alongside another CiU successor.
    if (core.id !== "pdcat") core.seats -= Number(Q.pdcat_parlament_s) || 0;
    var separate = row("pdcat", ["pdcat"], 2, 2);
    separate.hidden = core.id === "pdcat";
    if (separate.hidden) separate.seats = 0;
    var joint = row("joint", ["jxsi", "jxcat"], 0, 1);
    joint.isJoint = true;
    var rows = [
      joint,
      core,
      row("erc", ["erc"], 1, 2),
      row("si", ["si"], 1, 3),
      row("cup", ["cup"], 1, 4),
      row("icv", ["icv", "csqp", "cecp", "ecp"], 2, 1),
      separate,
      row("unio", ["unio"], 2, 3),
      row("pxc", ["pxc", "fnc"], 2, 4),
      row("cs", ["cs"], 3, 1),
      row("psc", ["psc"], 3, 2),
      row("ppc", ["ppc"], 3, 3),
      row("vox", ["vox"], 3, 4),
    ];
    [core, rows[2], rows[4]].forEach(function (entry) {
      entry.member = false;
      entry.group = "joint";
      entry.canJoin = true;
      entry.contribution = 0;
    });
    var config = { chamber: "parlament", names: names, rows: rows };
    if (joint.seats > 0) {
      [core, rows[2], rows[4]].forEach(function (entry) {
        if (entry.slot === "ciu" || Q[entry.id + "_in_" + joint.id])
          setBuilderMember(config, entry, true);
      });
    }
    joint.disabled = false;
    syncCoalitionBuilder(config);
    return config;
  }

  function setBuilderMember(config, row, member) {
    if (!!row.member === member) return;
    var joint = config.rows.find(function (entry) {
      return entry.slot === (row.group || "joint");
    });
    if (member) {
      row.contribution = row.seats;
      joint.seats += row.seats;
      row.seats = 0;
    } else {
      row.seats = Math.min(row.contribution || 0, joint.seats);
      joint.seats -= row.seats;
      row.contribution = 0;
    }
    row.member = member;
  }

  function clearCoalition(config, group) {
    config.rows
      .filter(function (entry) {
        return (entry.group || entry.slot) === group.slot;
      })
      .forEach(function (entry) {
        if (entry.member) {
          entry.seats = Math.max(0, entry.contribution || 0);
          entry.member = false;
        }
        entry.contribution = 0;
        entry.autoRequired = false;
      });
    group.seats = 0;
    group.disabled = true;
  }

  // Keep dependent slots consistent after every draft change. Hidden ballots
  // merge into their family carrier so identity switches cannot lose seats.
  function syncCoalitionBuilder(config) {
    var rows = config.rows;
    var joint = rows.find(function (row) {
      return row.slot === "joint";
    });
    var core = rows.find(function (row) {
      return row.slot === "ciu";
    });
    var erc = rows.find(function (row) {
      return row.slot === "erc";
    });
    var si = rows.find(function (row) {
      return row.slot === "si";
    });
    var pdcat = rows.find(function (row) {
      return row.slot === "pdcat";
    });
    var unio = rows.find(function (row) {
      return row.slot === "unio";
    });
    joint.hidden = false;
    core.canJoin = true;
    if (si) {
      si.group = "joint";
      setBuilderMember(config, si, !!erc.member);
    }
    var jointActive =
      joint.seats > 0 ||
      rows.some(function (row) {
        return (
          row.member && row.autoRequired && (row.group || "joint") === "joint"
        );
      });
    if (!core.disabled) setBuilderMember(config, core, jointActive);
    pdcat.hidden =
      core.id === "pdcat" || (core.id !== "junts" && joint.id !== "jxcat");
    unio.hidden = core.id === "ciu";
    [pdcat, unio].forEach(function (row) {
      if (!row.hidden || !row.seats) return;
      if (core.member) {
        joint.seats += row.seats;
        core.contribution = (core.contribution || 0) + row.seats;
      } else core.seats += row.seats;
      row.seats = 0;
    });
    if (config.chamber === "congreso") {
      var up = rows.find(function (row) {
        return row.slot === "up";
      });
      if (!up.disabled) {
        var upRequired =
          up.seats > 0 ||
          rows.some(function (row) {
            return (
              row.member && row.autoRequired && (row.group || "joint") === "up"
            );
          });
        ["iu", "podemos"].forEach(function (slot) {
          setBuilderMember(
            config,
            rows.find(function (row) {
              return row.slot === slot;
            }),
            upRequired,
          );
        });
      }
      var mpais = rows.find(function (row) {
        return row.slot === "mpais";
      });
      mpais.hidden = up.seats <= 0;
      if (mpais.hidden && mpais.seats) {
        rows.find(function (row) {
          return row.slot === "podemos";
        }).seats += mpais.seats;
        mpais.seats = 0;
      }
    }
  }

  function switchCoalitionBuilderParty(Q, slot, direction) {
    var config = Q.playground_coalitions;
    if (!config || (direction !== -1 && direction !== 1)) return false;
    var row = config.rows.find(function (entry) {
      return entry.slot === slot;
    });
    if (!row || row.variants.length < 2) return false;
    var index = row.variants.indexOf(row.id);
    row.id =
      row.variants[
        (index + direction + row.variants.length) % row.variants.length
      ];
    row.name = config.names[row.id];
    syncCoalitionBuilder(config);
    return true;
  }

  function toggleCoalitionBuilderMember(Q, slot) {
    var config = Q.playground_coalitions;
    if (!config) return false;
    var row = config.rows.find(function (entry) {
      return entry.slot === slot;
    });
    if (!row || !row.canJoin) return false;
    if (
      (row.slot === "ciu" || row.slot === "iu" || row.slot === "podemos") &&
      row.member
    ) {
      var group = config.rows.find(function (entry) {
        return entry.slot === row.group;
      });
      if (group) {
        clearCoalition(config, group);
        syncCoalitionBuilder(config);
        return true;
      }
    }
    if (row.member) row.autoRequired = false;
    setBuilderMember(config, row, !row.member);
    if (row.member) {
      if (row.slot === "ciu") {
        row.autoRequired = true;
      } else if (row.slot === "iu" || row.slot === "podemos") {
        row.autoRequired = true;
        var requiredPeer = config.rows.find(function (entry) {
          return entry.slot === (row.slot === "iu" ? "podemos" : "iu");
        });
        requiredPeer.autoRequired = true;
        setBuilderMember(config, requiredPeer, true);
      } else if (row.slot === "cup") {
        var core = config.rows.find(function (entry) {
          return entry.slot === "ciu";
        });
        var erc = config.rows.find(function (entry) {
          return entry.slot === "erc";
        });
        core.autoRequired = true;
        erc.autoRequired = true;
        setBuilderMember(config, core, true);
        setBuilderMember(config, erc, true);
      } else if (
        row.group === "up" &&
        ["bng", "compromis", "mes"].indexOf(row.slot) !== -1
      ) {
        ["iu", "podemos"].forEach(function (slot) {
          var required = config.rows.find(function (entry) {
            return entry.slot === slot;
          });
          required.autoRequired = true;
          setBuilderMember(config, required, true);
        });
      }
    }
    syncCoalitionBuilder(config);
    return true;
  }

  function editCoalitionBuilder(Q, id, seats) {
    var config = Q.playground_coalitions;
    if (
      !config ||
      !Number.isSafeInteger(seats) ||
      seats < 0 ||
      seats > Q.playground_coalitions_needed_seats
    )
      return false;
    var row = config.rows.find(function (entry) {
      return entry.id === id && !entry.hidden;
    });
    if (!row || row.member) return false;
    Q.playground_coalitions_set_seats += seats - row.seats;
    row.seats = seats;
    if (row.isJoint && seats > 0) row.disabled = false;
    if (row.isJoint && seats === 0) {
      config.rows
        .filter(function (entry) {
          return (entry.group || "joint") === row.slot;
        })
        .forEach(function (entry) {
          entry.member = false;
          entry.contribution = 0;
          entry.autoRequired = false;
        });
    }
    syncCoalitionBuilder(config);
    return true;
  }

  function applyCoalitionBuilder(Q) {
    var config = Q.playground_coalitions;
    if (
      !config ||
      Q.playground_coalitions_set_seats !== Q.playground_coalitions_needed_seats
    )
      return;
    Object.keys(config.names).forEach(function (id) {
      Q[id + "_" + config.chamber + "_s"] = 0;
    });
    if (config.chamber === "parlament") {
      ["jxsi", "jxcat"].forEach(function (list) {
        ["ciu", "cdc", "dl", "pdcat", "junts", "erc", "si", "cup"].forEach(
          function (id) {
            Q[id + "_in_" + list] = false;
          },
        );
        Q["player_in_" + list] = false;
        Q[list + "_formed"] = false;
        Q[list + "_dissolved"] = true;
      });
      var joint = config.rows[0];
      if (joint.seats > 0) {
        Q[joint.id + "_formed"] = true;
        Q[joint.id + "_dissolved"] = false;
        config.rows
          .filter(function (row) {
            return row.member;
          })
          .forEach(function (row) {
            Q[row.id + "_in_" + joint.id] = true;
            if (row.id === Q.player_party) Q["player_in_" + joint.id] = true;
          });
      }
      var core = config.rows.find(function (row) {
        return row.slot === "ciu";
      });
      var pdcat = config.rows.find(function (row) {
        return row.slot === "pdcat";
      });
      Q.pdcat_split = core.id !== "pdcat" && pdcat.seats > 0;
    }
    if (config.chamber === "congreso") {
      var up = config.rows.find(function (row) {
        return row.slot === "up";
      });
      var joint = config.rows.find(function (row) {
        return row.slot === "joint";
      });
      Q.iu_in_up = up.seats > 0;
      ["bng", "compromis", "mes"].forEach(function (slot) {
        Q["spa_" + slot + "_active"] = !config.rows.find(function (row) {
          return row.slot === slot;
        }).member;
      });
      Q.spa_mpais_active =
        config.rows.find(function (row) {
          return row.slot === "mpais";
        }).seats > 0;
      ["jxsi", "jxcat"].forEach(function (list) {
        Q[list + "_in_congreso"] = joint.id === list && joint.seats > 0;
        ["ciu", "cdc", "dl", "pdcat", "junts", "erc", "cup"].forEach(
          function (id) {
            Q[id + "_in_" + list] = false;
          },
        );
        Q["player_in_" + list] = false;
      });
      if (joint.seats > 0) {
        config.rows
          .filter(function (row) {
            return row.group === "joint" && row.member;
          })
          .forEach(function (row) {
            Q[row.id + "_in_" + joint.id] = true;
            if (row.id === Q.player_party) Q["player_in_" + joint.id] = true;
          });
      }
    }
    config.rows
      .filter(function (row) {
        return !row.hidden;
      })
      .forEach(function (row) {
        Q[row.id + "_" + config.chamber + "_s"] = row.member ? 0 : row.seats;
      });
  }

  var api = {
    coalitionBuilderConfig: coalitionBuilderConfig,
    editCoalitionBuilder: editCoalitionBuilder,
    switchCoalitionBuilderParty: switchCoalitionBuilderParty,
    toggleCoalitionBuilderMember: toggleCoalitionBuilderMember,
    applyCoalitionBuilder: applyCoalitionBuilder,
    congresoResultsMap: congresoResultsMap,
    localResultsMap: localResultsMap,
    congresoPartyTour: congresoPartyTour,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else window.RTI_ELECTION_WIDGETS = api;
})();
