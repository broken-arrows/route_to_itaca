const tooltipList = [
  /* PARTIES */
  {
    searchString: ["CiU", "ciu"],
    mainText: "Convergence and Union",
    subText: "Convergència i Unió",
    img: "img/parties/logo_ciu.png",
    ledBy: "ciu_leader",
    ideology: "ciu_ideology",
  },
  {
    searchString: ["CDC", "cdc"],
    mainText: "Democratic Convergence of Catalonia",
    subText: "Convergència Democràtica de Catalunya",
    img: "img/parties/logo_cdc.png",
    ledBy: "cdc_leader",
    ideology: "cdc_ideology",
  },
  {
    searchString: ["UDC", "udc", "unio"],
    mainText: "Democratic Union of Catalonia",
    subText: "Unió Democràtica de Catalunya",
    img: "img/parties/logo_unio.png",
    ledBy: "unio_leader",
    ideology: "unio_ideology",
  },
  {
    searchString: ["ERC", "erc"],
    mainText: "Republican Left of Catalonia",
    subText: "Esquerra Republicana de Catalunya",
    img: "img/parties/logo_erc.svg",
    ledBy: "erc_leader",
    ideology: "erc_ideology",
  },
  {
    searchString: ["CUP", "cup"],
    mainText: "Popular Unity Candidacy",
    subText: "Candidatura d'Unitat Popular",
    img: "img/parties/logo_cup.svg",
    ledBy: "cup_leader",
    ideology: "cup_ideology",
  },
  {
    searchString: ["JxSí", "JxSi", "jxsi"],
    mainText: 'Together for the "Yes"',
    subText: "Junts pel Sí",
    img: "img/parties/logo_jxsi.png",
    ledBy: "jxsi_leader",
    ideology: "jxsi_ideology",
  },
  {
    searchString: ["PDeCAT", "pdcat"],
    mainText: "Catalan European Democratic Party",
    subText: "Partit Demòcrata Europeu Català",
    img: "img/parties/logo_pdecat.svg",
    ledBy: "pdcat_leader",
    ideology: "pdcat_ideology",
  },
  {
    searchString: ["JxCat", "jxcat"],
    mainText: "Together for Catalonia",
    subText: "Junts per Catalunya",
    img: "img/parties/logo_jxcat.svg",
    ledBy: "jxcat_leader",
    ideology: "jxcat_ideology",
  },
  {
    searchString: ["Junts", "junts"],
    mainText: "Together",
    subText: "Junts",
    img: "img/parties/logo_junts.png",
    ledBy: "junts_leader",
    ideology: "junts_ideology",
  },
  {
    searchString: ["Cs", "cs", "Ciutadans"],
    mainText: "Citizens - Party of the Citizenry",
    subText: "Ciutadans - Partido de la Ciudadanía",
    img: "img/parties/logo_cs.svg",
    ledBy: "cs_leader",
    ideology: "cs_ideology",
  },
  {
    searchString: ["csspa", "cs_spa", "Ciudadanos"],
    mainText: "Citizens - Party of the Citizenry",
    subText: "Ciudadanos - Partido de la Ciudadanía",
    img: "img/parties/logo_cs.svg",
    ledBy: "csspa_leader",
    ideology: "cs_ideology",
  },
  {
    searchString: ["PPC", "ppc"],
    mainText: "People's Party of Catalonia",
    subText: "Partit Popular de Catalunya",
    img: "img/parties/logo_pp.png",
    ledBy: "ppc_leader",
    ideology: "ppc_ideology",
  },
  {
    searchString: ["PSC", "psc"],
    mainText: "Socialist Party of Catalonia",
    subText: "Partit dels Socialistes de Catalunya",
    img: "img/parties/logo_psc.svg",
    ledBy: "psc_leader",
    ideology: "psc_ideology",
  },
  {
    searchString: ["ICV-EUiA", "icv", "ICV"],
    mainText: "Initiative for Catalonia Greens - United and Alternative Left",
    subText: "Iniciativa per Catalunya Verds - Esquerra Unida i Alternativa",
    img: "img/parties/logo_icv.svg",
    ledBy: "icv_leader",
    ideology: "icv_ideology",
  },
  {
    searchString: ["CSQP", "csqp"],
    mainText: "Catalonia Yes We Can",
    subText: "Catalunya Sí que es Pot",
    img: "img/parties/logo_csqp.png",
    ledBy: "csqp_leader",
    ideology: "csqp_ideology",
  },
  {
    searchString: ["CECP", "cecp"],
    mainText: "Catalonia In Common We Can",
    subText: "Catalunya En Comú Podem",
    img: "img/parties/logo_cecp.svg",
    ledBy: "cecp_leader",
    ideology: "cecp_ideology",
  },
  {
    searchString: ["ECP", "ecp"],
    mainText: "In Common We Can",
    subText: "En Comú Podem",
    img: "img/parties/logo_ecp.svg",
    ledBy: "ecp_leader",
    ideology: "ecp_ideology",
  },
  {
    searchString: ["VOX", "Vox", "vox"],
    mainText: "VOX",
    subText: "VOX",
    img: "img/parties/logo_vox.svg",
    ledBy: "vox_leader",
    ideology: "vox_ideology",
  },
  {
    searchString: ["PxC", "pxc"],
    mainText: "Platform for Catalonia",
    subText: "Plataforma per Catalunya",
    img: "img/parties/logo_pxc.gif",
    ledBy: "pxc_leader",
    ideology: "pxc_ideology",
  },
  {
    searchString: ["FNC", "fnc"],
    mainText: "Catalan National Front",
    subText: "Front Nacional de Catalunya",
    img: "img/parties/logo_fnc.png",
    ledBy: "fnc_leader",
    ideology: "fnc_ideology",
  },
  {
    searchString: ["PP", "pp"],
    mainText: "People's Party",
    subText: "Partido Popular",
    img: "img/parties/logo_pp.png",
    ledBy: "pp_leader",
    ideology: "pp_ideology",
  },
  {
    searchString: ["PSOE", "psoe"],
    mainText: "Spanish Socialist Workers' Party",
    subText: "Partido Socialista Obrero Español",
    img: "img/parties/logo_psoe.png",
    ledBy: "psoe_leader",
    ideology: "psoe_ideology",
  },
  {
    searchString: ["Podemos", "podemos"],
    mainText: "We Can",
    subText: "Podemos",
    img: "img/parties/logo_podemos.svg",
    ledBy: "podemos_leader",
    ideology: "podemos_ideology",
  },
  {
    searchString: ["IU", "iu"],
    mainText: "United Left - Popular Unity",
    subText: "Izquierda Unida - Unidad Popular",
    img: "img/parties/logo_iu.svg",
    ledBy: "iu_leader",
    ideology: "iu_ideology",
  },
  {
    searchString: ["DL", "DiL", "dil", "dl"],
    mainText: "Democracy and Liberty",
    subText: "Democràcia i Llibertat",
    img: "img/parties/logo_dl.png",
    ledBy: "dl_leader",
    ideology: "dl_ideology",
  },
  {
    searchString: ["EAJ-PNV", "PNV", "pnv"],
    mainText: "Basque Nationalist Party",
    subText: "Euzko Alderdi Jeltzalea - Partido Nacionalista Vasco",
    img: "img/parties/logo_pnv.svg",
    ledBy: "pnv_leader",
    ideology: "pnv_ideology",
  },
  {
    searchString: ["EH Bildu", "ehbildu", "bildu"],
    mainText: "Basque Country Unite",
    subText: "Euskal Herria Bildu",
    img: "img/parties/logo_ehbildu.png",
    ledBy: "ehbildu_leader",
    ideology: "ehbildu_ideology",
  },
  {
    searchString: ["CC-PNC", "cc-pnc", "cc"],
    mainText: "Canarian Coalition - Canarian Nationalist Party",
    subText: "Coalición Canaria - Partido Nacionalista Canario",
    img: "img/parties/logo_ccpnc.svg",
    ledBy: "ccpnc_leader",
    ideology: "ccpnc_ideology",
  },
  {
    searchString: ["Compromís", "compromis"],
    mainText: "Commitment Coalition",
    subText: "Coalició Compromís",
    img: "img/parties/logo_compromis.svg",
    ledBy: "compromis_leader",
    ideology: "compromis_ideology",
  },
  {
    searchString: ["MÉS", "MES", "mes"],
    mainText: "More for Mallorca",
    subText: "Més per Mallorca",
    img: "img/parties/logo_mes.png",
    ledBy: "mes_leader",
    ideology: "mes_ideology",
  },
  {
    searchString: ["BNG", "bng"],
    mainText: "Galician Nationalist Bloc",
    subText: "Bloque Nacionalista Galego",
    img: "img/parties/logo_bng.svg",
    ledBy: "bng_leader",
    ideology: "bng_ideology",
  },
  {
    searchString: ["NOS", "NÓS", "nos"],
    mainText: "Nós - Popular Unity",
    subText: "Nós - Unidade Popular",
    img: "img/parties/logo_nos.jpg",
    ledBy: "nos_leader",
    ideology: "nos_ideology",
  },
  {
    searchString: ["Amaiur", "amaiur"],
    mainText: "Amaiur",
    subText: "Amaiur",
    img: "img/parties/logo_amaiur.svg",
    ledBy: "amaiur_leader",
    ideology: "amaiur_ideology",
  },
  {
    searchString: ["UPyD", "upyd"],
    mainText: "Union, Progress, and Democracy",
    subText: "Unión Progreso y Democracia",
    img: "img/parties/logo_upyd.svg",
    ledBy: "upyd_leader",
    ideology: "upyd_ideology",
  },
  {
    searchString: ["Más País", "mpais"],
    mainText: "More Country",
    subText: "Más País",
    img: "img/parties/logo_mpais.svg",
    ledBy: "mpais_leader",
    ideology: "mpais_ideology",
  },
  {
    searchString: ["FAC", "fac"],
    mainText: "Asturian Citizen Forum",
    subText: "Foro Asturias de Ciudadaions",
    img: "img/parties/logo_fac.svg",
    ledBy: "fac_leader",
    ideology: "fac_ideology",
  },
  {
    searchString: ["PRC", "prc"],
    mainText: "Regionalist Party of Cantabria",
    subText: "Partido Regionalista de Cantabria",
    img: "img/parties/logo_prc.png",
    ledBy: "prc_leader",
    ideology: "prc_ideology",
  },
  {
    searchString: ["UP"],
    mainText: "United We Can",
    subText: "Unidas Podemos",
    img: "img/parties/logo_up.svg",
    ledBy: "up_leader",
    ideology: "up_ideology",
  },
  {
    searchString: ["FR", "fr"],
    mainText: "Republican Front",
    subText: "Front Republicà",
    img: "img/parties/logo_fr.svg",
    ledBy: "fr_leader",
    ideology: "fr_ideology",
  },
  {
    searchString: ["GBai", "gbai"],
    mainText: "Yes to the Future",
    subText: "Geroa Bai",
    img: "img/parties/logo_gbai.svg",
    ledBy: "gbai_leader",
    ideology: "gbai_ideology",
  },
  {
    searchString: ["NA+", "na+", "nsuma"],
    mainText: "Sum Navarre",
    subText: "Navarra Suma",
    img: "img/parties/logo_nsuma.png",
    ledBy: "nsuma_leader",
    ideology: "nsuma_ideology",
  },
  {
    searchString: ["UPN", "upn"],
    mainText: "Navarrese People's Union",
    subText: "Unión del Pueblo Navarro",
    img: "img/parties/logo_upn.svg",
    ledBy: "upn_leader",
    ideology: "upn_ideology",
  },
  {
    searchString: ["¡TE!", "te", "texiste"],
    mainText: "Teruel Exists!",
    subText: "¡Teruel Existe!",
    img: "img/parties/logo_te.jpg",
    ledBy: "te_leader",
    ideology: "te_ideology",
  },
  {
    searchString: ["SI"],
    mainText: "Catalan Solidarity for Independence",
    subText: "Solidaritat Catalana per la Independència",
    img: "img/parties/logo_si.jpg",
    ledBy: "si_leader",
    ideology: "si_ideology",
  },
  {
    searchString: ["BComu", "bcomu", "Barcelona en Comú", "BComú"],
    mainText: "Barcelona in Common",
    subText: "Barcelona en Comú",
    img: "img/parties/logo_bcomu.png",
    ledBy: "bcomu_leader",
    ideology: "bcomu_ideology",
  },
  {
    searchString: ["ppbcn", "pp_bcn"],
    mainText: "People's Party (Barcelona)",
    subText: "Partit Popular",
    img: "img/parties/logo_pp.png",
    ledBy: "ppbcn_leader",
    ideology: "ppc_ideology",
  },
  {
    searchString: ["csbcn", "cs_bcn"],
    mainText: "Citizens - Party of the Citizenry (Barcelona)",
    subText: "Ciutadans - Partido de la Ciudadanía",
    img: "img/parties/logo_cs.svg",
    ledBy: "csbcn_leader",
    ideology: "cs_ideology",
  },
  {
    searchString: ["pscbcn", "psc_bcn"],
    mainText: "Socialist Party of Catalonia (Barcelona)",
    subText: "Partit dels Socialistes de Catalunya",
    img: "img/parties/logo_psc.svg",
    ledBy: "pscbcn_leader",
    ideology: "psc_ideology",
  },
  {
    searchString: ["ercbcn", "erc_bcn"],
    mainText: "Republican Left of Catalonia (Barcelona)",
    subText: "Esquerra Republicana de Catalunya",
    img: "img/parties/logo_erc.svg",
    ledBy: "ercbcn_leader",
    ideology: "erc_ideology",
  },
  {
    searchString: ["cupbcn", "cup_bcn"],
    mainText: "Popular Unity Candidacy (Barcelona)",
    subText: "Candidatura d'Unitat Popular",
    img: "img/parties/logo_cup.svg",
    ledBy: "cupbcn_leader",
    ideology: "cup_ideology",
  },
  {
    searchString: ["cdcbcn", "cdc_bcn"],
    mainText: "Democratic Convergence of Catalonia (Barcelona)",
    subText: "Convergència Democràtica de Catalunya",
    img: "img/parties/logo_cdc.png",
    ledBy: "cdcbcn_leader",
    ideology: "cdc_ideology",
  },
  {
    searchString: ["dlbcn", "dl_bcn"],
    mainText: "Democracy and Liberty (Barcelona)",
    subText: "Democràcia i Llibertat",
    img: "img/parties/logo_dl.png",
    ledBy: "dlbcn_leader",
    ideology: "dl_ideology",
  },
  {
    searchString: ["jxsibcn", "jxsi_bcn"],
    mainText: 'Together for the "Yes" (Barcelona)',
    subText: "Junts pel Sí",
    img: "img/parties/logo_jxsi.png",
    ledBy: "jxsibcn_leader",
    ideology: "jxsi_ideology",
  },
  {
    searchString: ["pdcatbcn", "pdcat_bcn", "pdecatbcn", "pdecat_bcn"],
    mainText: "Catalan European Democratic Party (Barcelona)",
    subText: "Partit Demòcrata Europeu Català",
    img: "img/parties/logo_pdecat.svg",
    ledBy: "pdcatbcn_leader",
    ideology: "pdcat_ideology",
  },
  {
    searchString: ["jxcatbcn", "jxcat_bcn"],
    mainText: "Together for Catalonia (Barcelona)",
    subText: "Junts per Catalunya",
    img: "img/parties/logo_jxcat.svg",
    ledBy: "jxcatbcn_leader",
    ideology: "jxcat_ideology",
  },
  {
    searchString: ["juntsbcn", "junts_bcn"],
    mainText: "Together (Barcelona)",
    subText: "Junts",
    img: "img/parties/logo_junts.png",
    ledBy: "juntsbcn_leader",
    ideology: "junts_ideology",
  },
  {
    searchString: ["ciubcn", "ciu_bcn"],
    mainText: "Convergence and Union (Barcelona)",
    subText: "Convergència i Unió",
    img: "img/parties/logo_ciu.png",
    ledBy: "ciubcn_leader",
    ideology: "ciu_ideology",
  },
  {
    searchString: ["primariesbcn"],
    mainText: "Barcelona is Capital - Primaries",
    subText: "Primàries - Barcelona és Capital",
    img: "img/parties/logo_primaries.svg",
    ledBy: "primaries_leader",
    ideology: "primaries_ideology",
  },
  {
    searchString: ["ua-psc", "UA", "UA-PSC"],
    mainText: "Aran Unity - Socialist Party of Catalonia",
    subText: "Unitat d'Aran - Partit dels Socialistes de Catalunya",
    img: "img/parties/logo_ua.svg",
    ledBy: "ua_psc_leader",
    ideology: "ua_psc_ideology",
  },
  {
    searchString: ["cda-pna", "CDA-PNA", "CDA", "cda"],
    mainText: "Aranese Democratic Convergence - Aranese Nationalist Party",
    subText: "Convergéncia Democratica Aranesa - Partit Nacionalista Aranés",
    img: "img/parties/logo_cda.svg",
    ledBy: "cda_pna_leader",
    ideology: "cda_pna_ideology",
  },
  {
    searchString: ["txt"],
    mainText: "Everything for Terrassa",
    subText: "Tot per Terrassa",
    img: "img/parties/logo_txt.png",
    ledBy: "txt_leader",
    ideology: "txt_ideology",
  },
  {
    searchString: ["ETA"],
    mainText: "Basque Homeland and Liberty",
    subText: "Euskadi Ta Askatasuna",
    img: "img/parties/eta.webp",
    infoDesc: "Paramilitary Clandestine Organization",
    ideology: "Revolutionary Socialism, Basque Independence",
  },
  {
    searchString: ["ANC"],
    mainText: "Catalan National Assembly",
    subText: "Assemblea Nacional Catalana",
    img: "img/parties/anc.webp",
    infoDesc: "NGO, Political Organization",
    ideology: "Big tent single-issue, Independence",
  },
  {
    searchString: ["Omnium Cultural", "Omnium"],
    mainText: "Òmnium Cultural",
    subText: "Òmnium Cultural",
    img: "img/parties/logo_omnium.png",
    infoDesc: "NGO, Cultural Association",
    ideology: "Transversal Catalanism, Independence",
  },
  /* LEADERS */
  {
    searchString: ["Lluís Companys"],
    mainText: "Lluís Companys i Jover",
    img: "img/erc/lluiscompanys.jpg",
    ideology: "Left-wing Republicanism, Catalan Nationalism",
    allegiances: (Q) => {
      return ["<span style='color: var(--erc)'>ERC</span>"];
    },
  },
  {
    searchString: ["Francisco Franco", "Franco"],
    mainText: "Francisco Franco Bahamonde",
    img: "img/other_leaders/franco.jpg",
    ideology: "Fascism, National Catholicism",
    allegiances: (Q) => {
      return ["<span style='color: #555555'>FET y de las JONS</span>"];
    },
  },
  {
    searchString: ["Jordi Pujol"],
    mainText: "Jordi Pujol i Soley",
    img: "img/ciu/jordipujol.jpg",
    ideology: "Center-right Liberalism, Catalan Nationalism",
    allegiances: (Q) => {
      return [
        "<span style='color: var(--ciu)'>CDC</span>",
        "<span style='color: var(--ciu)'>CiU</span>",
      ];
    },
  },
  {
    searchString: ["Artur Mas", "artur mas", "Artur Mas i Gavarró"],
    mainText: "Artur Mas i Gavarró",
    img: "img/ciu/arturmas.jpg",
    ideology: "Center-right Liberalism, Pact-Pragmatism",
    allegiances: (Q) => {
      let list = [
        "<span style='color: var(--cdc)'>CDC</span>",
        "<span style='color: var(--ciu)'>CiU</span>",
      ];
      if (Q.dl_formed) list.push("<span style='color: var(--dl)'>DL</span>");
      if (Q.jxsi_formed)
        list.push("<span style='color: var(--jxsi)'>JxSí</span>");
      if (Q.pdcat_formed)
        list.push("<span style='color: var(--pdcat)'>PDeCAT</span>");
      if (Q.jxcat_formed && !Q.mas_ousted)
        list.push("<span style='color: var(--jxcat)'>JxCat</span>");
      if (Q.junts_formed && !Q.mas_ousted)
        list.push("<span style='color: var(--junts)'>Junts</span>");
      return list;
    },
  },
  {
    searchString: [
      "Carles Puigdemont",
      "carles puigdemont",
      "Carles Puigdemont i Casamajó",
    ],
    mainText: "Carles Puigdemont i Casamajó",
    img: "img/ciu/carlespuigdemont.jpg",
    ideology: "Center-right Liberalism, Unilateralism",
    allegiances: (Q) => {
      let list = [
        "<span style='color: var(--cdc)'>CDC</span>",
        "<span style='color: var(--ciu)'>CiU</span>",
      ];
      if (Q.dl_formed) list.push("<span style='color: var(--dl)'>DL</span>");
      if (Q.jxsi_formed)
        list.push("<span style='color: var(--jxsi)'>JxSí</span>");
      if (Q.pdcat_formed && !Q.pdcat_split) {
        list.push("<span style='color: var(--pdcat)'>PDeCAT</span>");
      } else if (Q.pdcat_formed) {
        list.push("<span style='color: var(--pdcat)'>PDeCAT</span> (former)");
      }
      if (Q.jxcat_formed)
        list.push("<span style='color: var(--jxcat)'>JxCat</span>");
      if (Q.junts_formed)
        list.push("<span style='color: var(--junts)'>Junts</span>");
      return list;
    },
  },
  {
    searchString: ["Josep Antoni Duran i Lleida", "Duran i Lleida"],
    mainText: "Josep Antoni Duran i Lleida",
    img: "img/ciu/jaduranilleida.jpg",
    ideology: "Christian Democracy, Pact-Pragmatism",
    allegiances: (Q) => {
      return [
        "<span style='color: var(--unio)'>UDC</span>",
        "<span style='color: var(--ciu)'>CiU</span>",
      ];
    },
  },
  {
    searchString: ["Oriol Junqueras", "Oriol Junqueras i Vies"],
    mainText: "Oriol Junqueras i Vies",
    img: "img/erc/junqueras.jpg",
    ideology: "Left-wing Republicanism, Independence",
    allegiances: (Q) => {
      let list = ["<span style='color: var(--erc)'>ERC</span>"];
      if (Q.jxsi_formed)
        list.push("<span style='color: var(--jxsi)'>JxSí</span>");
      if (Q.jxcat_formed && Q.erc_in_jxcat)
        list.push("<span style='color: var(--jxcat)'>JxCat</span>");
      return list;
    },
  },
  {
    searchString: ["David Fernàndez", "David Fernàndez i Ramos"],
    mainText: "David Fernàndez i Ramos",
    img: "img/cup/davidfernandez.jpg",
    ideology: "Anti-capitalism, Unilateralism",
    allegiances: (Q) => {
      return ["<span style='color: #b8a12b'>CUP</span>"];
    },
  },
  {
    searchString: ["Albert Rivera", "Albert Rivera Díaz"],
    mainText: "Albert Rivera Díaz",
    img: "img/other_leaders/albert_rivera.jpg",
    ideology: "Neo-liberalism, Centralism",
    allegiances: (Q) => {
      return ["<span style='color: var(--cs)'>Cs</span>"];
    },
  },
  {
    searchString: ["Alfons López Tena"],
    mainText: "Alfons López Tena",
    img: "img/other_leaders/alfons_lopeztena.jpg",
    ideology: "Centerist regeneration, Unilateralism",
    allegiances: (Q) => {
      return ["<span style='color: var(--si)'>SI</span>"];
    },
  },
  {
    searchString: ["Pere Navarro", "Pere Navarro i Morera"],
    mainText: "Pere Navarro i Morera",
    img: "img/other_leaders/pere_navarro.jpg",
    ideology: "Centrist Social Democracy, Federalism",
    allegiances: (Q) => {
      let list = ["<span style='color: var(--psc)'>PSC</span>"];
      if (Q.psc_leader != "Pere Navarro")
        list.push("<span style='color: var(--psoe)'>PSOE</span>");
      return list;
    },
  },
  {
    searchString: ["Alícia Sánchez-Camacho"],
    mainText: "Alícia Sánchez-Camacho Pérez",
    img: "img/other_leaders/alicia_sanchezcamacho.jpg",
    ideology: "Conservatism, Centralism",
    allegiances: (Q) => {
      return [
        "<span style='color: var(--pp)'>PP</span>",
        "<span style='color: var(--ppc)'>PPC</span>",
      ];
    },
  },
  {
    searchString: ["Joan Herrera", "Joan Herrera i Torres"],
    mainText: "Joan Herrera i Torres",
    img: "img/other_leaders/joan_herrera.jpg",
    ideology: "Green Left, Plurinationalism",
    allegiances: (Q) => {
      let list = ["<span style='color: var(--icv)'>ICV-EUiA</span>"];
      if (Q.csqp_formed)
        list.push("<span style='color: var(--csqp)'>CSQP</span>");
      return list;
    },
  },
  {
    searchString: ["Mariano Rajoy", "Rajoy", "rajoy"],
    mainText: "Mariano Rajoy Brey",
    img: "img/other_leaders/rajoy.jpg",
    ideology: "Conservatism, Centralism",
    allegiances: (Q) => {
      return ["<span style='color: var(--pp)'>PP</span>"];
    },
  },
  {
    searchString: ["Pedro Sánchez"],
    mainText: "Pedro Sánchez Pérez-Castejón",
    img: "img/other_leaders/pedro_sanchez.jpg",
    ideology: "Centrist Social Democracy, Reformist Autonomism",
    allegiances: (Q) => {
      return ["<span style='color: var(--psoe)'>PSOE</span>"];
    },
  },
  {
    searchString: ["Eduardo Madina"],
    mainText: "Eduardo Madina Muñoz",
    img: "img/other_leaders/madina.jpg",
    ideology: "Centrist Social Democracy, Republican Autonomism",
    allegiances: (Q) => {
      return [
        "<span style='color: var(--psc)'>PSE-EE</span>",
        "<span style='color: var(--psoe)'>PSOE</span>",
      ];
    },
  },
  {
    searchString: ["Susana Díaz", "Susana Diaz"],
    mainText: "Susana Díaz Pacheco",
    img: "img/other_leaders/susanadiaz.jpg",
    ideology: "Centrist Social Democracy, Autonomism",
    allegiances: (Q) => {
      return [
        "<span style='color: var(--psoe)'>PSOE-A</span>",
        "<span style='color: var(--psoe)'>PSOE</span>",
      ];
    },
  },
  {
    searchString: ["Felipe González"],
    mainText: "Felipe González Márquez ",
    img: "img/other_leaders/fgonzalez.jpg",
    ideology: "Centrist Social Democracy, Autonomism",
    allegiances: (Q) => {
      return ["<span style='color: var(--psoe)'>PSOE</span>"];
    },
  },
  {
    searchString: ["Alfredo Pérez Rubalcaba", "Rubalcaba"],
    mainText: "Alfredo Pérez Rubalcaba",
    img: "img/other_leaders/rubalcaba.jpg",
    ideology: "Centrist Social Democracy, Autonomism",
    allegiances: (Q) => {
      return ["<span style='color: var(--psoe)'>PSOE</span>"];
    },
  },
  {
    searchString: ["Joan Laporta", "Joan Laporta i Estruch"],
    mainText: "Joan Laporta i Estruch",
    img: "img/erc/laporta.jpg",
    ideology: "Centerist regeneration, Independence",
    allegiances: (Q) => {
      return [
        "<span style='color: var(--si)'>SI</span>",
        "<span style='color: var(--erc)'>ERC</span>",
      ];
    },
  },
  {
    searchString: ["Ada Colau"],
    mainText: "Ada Colau i Ballano",
    img: "img/other_leaders/ada_colau.jpg",
    ideology: "Anti-austerity Left, Plurinationalism",
    allegiances: (Q) => {
      let list = ["<span style='color: var(--bcomu)'>BComú</span>"];
      if (Q.csqp_formed)
        list.push("<span style='color: var(--csqp)'>CSQP</span>");
      if (Q.cecp_formed)
        list.push("<span style='color: var(--cecp)'>CECP</span>");
      if (Q.ecp_formed) list.push("<span style='color: var(--ecp)'>ECP</span>");
      return list;
    },
  },
  {
    searchString: ["Àngel Ros", "Angel Ros", "angel ros"],
    mainText: "Àngel Ros i Domingo",
    img: "img/other_leaders/angel_ros.jpg",
    ideology: "Centrist Social Democracy, Plurinationalism",
    allegiances: (Q) => {
      return ["<span style='color: var(--psc)'>PSC</span>"];
    },
  },
  {
    searchString: ["Montserrat Tura", "montserrat tura"],
    mainText: "Montserrat Tura i Camafreita",
    img: "img/other_leaders/montserrat_tura.jpg",
    ideology: "Social Democracy, Plurinationalism",
    allegiances: (Q) => {
      let list = [];
      if (
        !(Q.psc_implosion_countdown < 0) ||
        Q.psc_leader == "Montserrat Tura" ||
        Q.psc_leader == "Àngel Ros"
      ) {
        list.push("<span style='color: var(--psc)'>PSC</span>");
      } else {
        list.push("<span style='color: var(--psc)'>PSC</span> (former)");
        list.push("<span style='color: var(--indp)'>indp.</span>");
      }
      return list;
    },
  },
  {
    searchString: ["Ernest Maragall", "ernest maragall"],
    mainText: "Ernest Maragall i Mira",
    img: "img/erc/ernestmaragall.jpg",
    ideology: "Social Democracy, Pragmatic Independence",
    allegiances: (Q) => {
      let list = [];
      if (Q.psc_leader == "Montserrat Tura") {
        list.push("<span style='color: var(--psc)'>PSC</span>");
      } else {
        list.push("<span style='color: var(--psc)'>PSC</span> (former)");
      }
      if (Q.ernest_maragall_advisor_available) {
        list.push("<span style='color: var(--erc)'>ERC</span>");
      } else {
        list.push("<span style='color: var(--indp)'>indp.</span>");
      }
      return list;
    },
  },
  {
    searchString: ["Pasqual Maragall", "pasqual maragall"],
    mainText: "Pasqual Maragall i Mira",
    img: "img/other_leaders/pasqual_maragall.jpg",
    ideology: "Social Democracy, Pragmatic Independence",
    allegiances: (Q) => {
      return ["<span style='color: var(--psc)'>PSC</span>"];
    },
  },
  {
    searchString: ["Núria Parlon", "Nuria Parlon", "nuria parlon"],
    mainText: "Núria Parlon i Gil",
    img: "img/other_leaders/nuria_parlon.jpg",
    ideology: "Centrist Social Democracy, Federalism",
    allegiances: (Q) => {
      if (Q.art155_ever) {
        return [
          "<span style='color: var(--psc)'>PSC</span> (former)",
          "<span style='color: var(--indp)'>indp.</span>",
        ];
      } else {
        return ["<span style='color: var(--psc)'>PSC</span>"];
      }
    },
  },
  {
    searchString: ["Miquel Iceta", "miquel iceta"],
    mainText: "Miquel Iceta i Llorens",
    img: "img/other_leaders/miquel_iceta.jpg",
    ideology: "Centrist Social Democracy, Autonomism",
    allegiances: (Q) => {
      return ["<span style='color: var(--psc)'>PSC</span>"];
    },
  },
  {
    searchString: ["Gabriel Rufian", "gabriel rufian", "Gabriel Rufián"],
    mainText: "Gabriel Rufian i Romero",
    img: "img/erc/rufian.png",
    ideology: "Left Social Democracy, Independence",
    allegiances: (Q) => {
      return ["<span style='color: var(--erc)'>ERC</span>"];
    },
  },
  {
    searchString: ["Soraya Sáenz de Santamaría", "soraya saenz de santamaria"],
    mainText: "Soraya Sáenz de Santamaría",
    img: "img/other_leaders/soraya.jpg",
    ideology: "Conservatism, Centralism",
    allegiances: (Q) => {
      return ["<span style='color: var(--pp)'>PP</span>"];
    },
  },
  {
    searchString: ["Raül Romeva", "Raul Romeva", "raul romeva"],
    mainText: "Raül Romeva i Rueda",
    img: "img/erc/romeva.jpg",
    ideology: "Social Democracy, Independence",
    allegiances: (Q) => {
      if (!Q.jxsi_formed) {
        return ["<span style='color: var(--icv)'>ICV-EUiA</span>"];
      } else {
        if (Q.jxcat_formed && Q.erc_in_jxcat) {
          return [
            "<span style='color: var(--icv)'>ICV-EUiA</span> (former)",
            "<span style='color: var(--jxsi)'>JxSí</span>",
            "<span style='color: var(--jxcat)'>JxCat</span>",
            "<span style='color: var(--erc)'>ERC</span>",
          ];
        } else if (Q.jxcat_formed && !Q.erc_in_jxcat) {
          return [
            "<span style='color: var(--icv)'>ICV-EUiA</span> (former)",
            "<span style='color: var(--jxsi)'>JxSí</span>",
            "<span style='color: var(--erc)'>ERC</span>",
          ];
        } else {
          return [
            "<span style='color: var(--icv)'>ICV-EUiA</span> (former)",
            "<span style='color: var(--jxsi)'>JxSí</span>",
          ];
        }
      }
    },
  },
  {
    searchString: ["Pablo Iglesias", "pablo iglesias"],
    mainText: "Pablo Iglesias Turrión",
    img: "img/other_leaders/pablo_iglesias.jpg",
    ideology: "Anti-austerity left, Plurinationalism",
    allegiances: (Q) => {
      if (Q.iu_in_up) {
        return [
          "<span style='color: var(--podemos)'>Podemos</span>",
          "<span style='color: var(--up)'>UP</span>",
        ];
      } else {
        return ["<span style='color: var(--podemos)'>Podemos</span>"];
      }
    },
  },
];

const colourList = [
  /* PARTIES */
  {
    words: ["CiU", "ciu"],
    colour: "var(--ciu)",
    style: "font-weight: bold;",
    transform: "CiU",
  },
  {
    words: ["CDC", "cdc"],
    colour: "var(--cdc)",
    style: "font-weight: bold;",
    transform: "CDC",
  },
  {
    words: ["UDC", "udc", "unio"],
    colour: "var(--unio)",
    style: "font-weight: bold;",
    transform: "UDC",
  },
  {
    words: ["ERC", "erc"],
    colour: "var(--erc)",
    style: "font-weight: bold;",
    transform: "ERC",
  },
  {
    words: ["CUP", "cup"],
    colour: "#b8a12b",
    style: "font-weight: bold;",
    transform: "CUP",
  },
  {
    words: ["JxSí", "jxsi"],
    colour: "var(--jxsi)",
    style: "font-weight: bold;",
    transform: "JxSí",
  },
  {
    words: ["JxCat", "jxcat"],
    colour: "var(--jxcat)",
    style: "font-weight: bold;",
    transform: "JxCat",
  },
  {
    words: ["PDeCAT", "pdcat"],
    colour: "var(--pdcat)",
    style: "font-weight: bold;",
    transform: "PDeCAT",
  },
  {
    words: ["Junts", "junts"],
    colour: "var(--junts)",
    style: "font-weight: bold;",
    transform: "Junts",
  },
  {
    words: ["Cs", "cs", "Ciutadans"],
    colour: "var(--cs)",
    style: "font-weight: bold;",
    transform: "Cs",
  },
  {
    words: ["csspa", "cs_spa", "Ciudadanos"],
    colour: "var(--cs)",
    style: "font-weight: bold;",
    transform: "Cs",
  },
  {
    words: ["PPC", "ppc"],
    colour: "var(--ppc)",
    style: "font-weight: bold;",
    transform: "PPC",
  },
  {
    words: ["PSC", "psc"],
    colour: "var(--psc)",
    style: "font-weight: bold;",
    transform: "PSC",
  },
  {
    words: ["ICV-EUiA", "icv"],
    colour: "var(--icv)",
    style: "font-weight: bold;",
    transform: "ICV-EUiA",
  },
  {
    words: ["ICV"],
    colour: "var(--icv)",
    style: "font-weight: bold;",
    transform: "ICV",
  },
  {
    words: ["CSQP", "csqp"],
    colour: "var(--csqp)",
    style: "font-weight: bold;",
    transform: "CSQP",
  },
  {
    words: ["CECP", "cecp"],
    colour: "var(--cecp)",
    style: "font-weight: bold;",
    transform: "CECP",
  },
  {
    words: ["ECP", "ecp"],
    colour: "var(--ecp)",
    style: "font-weight: bold;",
    transform: "ECP",
  },
  {
    words: ["VOX", "Vox", "vox"],
    colour: "var(--vox)",
    style: "font-weight: bold;",
    transform: "VOX",
  },
  {
    words: ["PxC", "pxc"],
    colour: "var(--pxc)",
    style: "font-weight: bold;",
    transform: "PxC",
  },
  {
    words: ["FNC", "fnc"],
    colour: "var(--fnc)",
    style: "font-weight: bold;",
    transform: "FNC",
  },
  {
    words: ["PP", "pp"],
    colour: "var(--pp)",
    style: "font-weight: bold;",
    transform: "PP",
  },
  {
    words: ["PSOE", "psoe"],
    colour: "var(--psoe)",
    style: "font-weight: bold;",
    transform: "PSOE",
  },
  {
    words: ["Podemos", "podemos"],
    colour: "var(--podemos)",
    style: "font-weight: bold;",
    transform: "Podemos",
  },
  {
    words: ["IU", "iu"],
    colour: "var(--iu)",
    style: "font-weight: bold;",
    transform: "IU",
  },
  {
    words: ["DL", "DiL", "dil", "dl"],
    colour: "var(--dl)",
    style: "font-weight: bold;",
    transform: "DL",
  },
  {
    words: ["EAJ-PNV", "PNV", "pnv"],
    colour: "var(--pnv)",
    style: "font-weight: bold;",
    transform: "EAJ-PNV",
  },
  {
    words: ["EH Bildu", "ehbildu", "bildu"],
    colour: "var(--ehbildu)",
    style: "font-weight: bold;",
    transform: "EH Bildu",
  },
  {
    words: ["CC-PNC", "cc-pnc", "cc"],
    colour: "var(--cc)",
    style: "font-weight: bold;",
    transform: "CC-PNC",
  },
  {
    words: ["Compromís", "compromis"],
    colour: "var(--compromis)",
    style: "font-weight: bold;",
    transform: "Compromís",
  },
  {
    words: ["MÉS", "MES", "mes"],
    colour: "var(--mes)",
    style: "font-weight: bold;",
    transform: "MÉS",
  },
  {
    words: ["BNG", "bng"],
    colour: "var(--bng)",
    style: "font-weight: bold;",
    transform: "BNG",
  },
  {
    words: ["NOS", "NÓS", "nos"],
    colour: "var(--nos)",
    style: "font-weight: bold;",
    transform: "NÓS",
  },
  {
    words: ["Amaiur", "amaiur"],
    colour: "var(--amaiur)",
    style: "font-weight: bold;",
    transform: "Amaiur",
  },
  {
    words: ["UPyD", "upyd"],
    colour: "var(--upyd)",
    style: "font-weight: bold;",
    transform: "UPyD",
  },
  {
    words: ["Más País", "mpais"],
    colour: "var(--mpais)",
    style: "font-weight: bold;",
    transform: "Más País",
  },
  {
    words: ["FAC", "fac"],
    colour: "var(--fac)",
    style: "font-weight: bold;",
    transform: "FAC",
  },
  {
    words: ["PRC", "prc"],
    colour: "var(--prc)",
    style: "font-weight: bold;",
    transform: "PRC",
  },
  {
    words: ["Sumar", "sumar"],
    colour: "var(--sumar)",
    style: "font-weight: bold;",
    transform: "Sumar",
  },
  {
    words: ["UP"],
    colour: "var(--up)",
    style: "font-weight: bold;",
    transform: "UP",
  },
  {
    words: ["FR", "fr"],
    colour: "var(--fr)",
    style: "font-weight: bold;",
    transform: "FR",
  },
  {
    words: ["GBai", "gbai"],
    colour: "var(--gbai)",
    style: "font-weight: bold;",
    transform: "GBai",
  },
  {
    words: ["UPN", "upn"],
    colour: "var(--upn)",
    style: "font-weight: bold;",
    transform: "UPN",
  },
  {
    words: ["NA+", "na+", "nsuma"],
    colour: "var(--nsuma)",
    style: "font-weight: bold;",
    transform: "NA+",
  },
  {
    words: ["¡TE!", "te", "texiste"],
    colour: "var(--te)",
    style: "font-weight: bold;",
    transform: "¡TE!",
  },
  {
    words: ["SI"],
    colour: "var(--si)",
    style: "font-weight: bold;",
    transform: "SI",
  },
  {
    words: ["BComu", "bcomu", "Barcelona en Comú", "BComú"],
    colour: "var(--bcomu)",
    style: "font-weight: bold;",
    transform: "BComú",
  },
  {
    words: ["ppbcn"],
    colour: "var(--ppc)",
    style: "font-weight: bold;",
    transform: "PP",
  },
  {
    words: ["pscbcn"],
    colour: "var(--psc)",
    style: "font-weight: bold;",
    transform: "PSC",
  },
  {
    words: ["csbcn"],
    colour: "var(--cs)",
    style: "font-weight: bold;",
    transform: "Cs",
  },
  {
    words: ["cupbcn"],
    colour: "#b8a12b",
    style: "font-weight: bold;",
    transform: "CUP",
  },
  {
    words: ["ercbcn"],
    colour: "var(--erc)",
    style: "font-weight: bold;",
    transform: "ERC",
  },
  {
    words: ["ciubcn"],
    colour: "var(--ciu)",
    style: "font-weight: bold;",
    transform: "CiU",
  },
  {
    words: ["cdcbcn"],
    colour: "var(--cdc)",
    style: "font-weight: bold;",
    transform: "CDC",
  },
  {
    words: ["uniobcn", "udcbcn"],
    colour: "var(--unio)",
    style: "font-weight: bold;",
    transform: "UDC",
  },
  {
    words: ["jxsi_bcn", "jxsibcn"],
    colour: "var(--jxsi)",
    style: "font-weight: bold;",
    transform: "JxSí",
  },
  {
    words: ["jxcat_bcn", "jxcatbcn"],
    colour: "var(--jxcat)",
    style: "font-weight: bold;",
    transform: "JxCat",
  },
  {
    words: ["pdcat_bcn", "pdcatbcn"],
    colour: "var(--pdcat)",
    style: "font-weight: bold;",
    transform: "PDeCAT",
  },
  {
    words: ["junts_bcn", "juntsbcn"],
    colour: "var(--junts)",
    style: "font-weight: bold;",
    transform: "Junts",
  },
  {
    words: ["dl_bcn", "dlbcn"],
    colour: "var(--dl)",
    style: "font-weight: bold;",
    transform: "DL",
  },
  {
    words: ["fncbcn"],
    colour: "var(--fnc)",
    style: "font-weight: bold;",
    transform: "FNC",
  },
  {
    words: ["pxcbcn"],
    colour: "var(--pxc)",
    style: "font-weight: bold;",
    transform: "PxC",
  },
  {
    words: ["vox_bcn", "voxbcn"],
    colour: "var(--vox)",
    style: "font-weight: bold;",
    transform: "VOX",
  },
  {
    words: ["primariesbcn", "primaries_bcn"],
    colour: "var(--primaries)",
    style: "font-weight: bold;",
    transform: "Primàries",
  },
  {
    words: ["ua-psc", "UA", "UA-PSC"],
    colour: "var(--ua-psc)",
    style: "font-weight: bold;",
    transform: "UA-PSC",
  },
  {
    words: ["cda-pna", "CDA-PNA"],
    colour: "var(--cda-pna)",
    style: "font-weight: bold;",
    transform: "CDA-PNA",
  },
  {
    words: ["txt"],
    colour: "var(--txt)",
    style: "font-weight: bold;",
    transform: "TxT",
  },
  {
    words: ["ETA"],
    colour: "#1e191a",
    style: "font-weight: bold;",
    transform: "ETA",
  },
  {
    words: ["ANC"],
    colour: "#010202",
    style: "font-weight: bold;",
    transform: "ANC",
  },
  {
    words: ["Omnium"],
    colour: "#e97202",
    style: "font-weight: bold;",
    transform: "Òmnium",
  },
  {
    words: ["Omnium Cultural"],
    colour: "#e97202",
    style: "font-weight: bold;",
    transform: "Òmnium Cultural",
  },
  {
    words: ["indp"],
    colour: "#464646",
    style: "font-weight: bold;",
  },
  /* LEADERS */
  {
    words: ["Lluís Companys", "Companys"],
    colour: "var(--erc)",
  },
  {
    words: ["Francisco Franco", "Franco"],
    colour: "#3e3e3e",
  },
  {
    words: ["Jordi Pujol"],
    colour: "var(--ciu)",
  },
  {
    words: ["Artur Mas", "Artur Mas i Gavarró"],
    colour: "var(--cdc)",
  },
  {
    words: ["Carles Puigdemont", "Carles Puigdemont i Casamajó"],
    colour: "var(--jxsi)",
  },
  {
    words: ["Raül Romeva", "Raul Romeva", "Raül Romeva i Rueda", "raul romeva"],
    transform: "Raül Romeva",
    colour: "var(--jxsi)",
  },
  {
    words: ["Oriol Junqueras", "Oriol Junqueras i Vies"],
    colour: "var(--erc)",
  },
  {
    words: ["David Fernàndez", "David Fernàndez i Ramos"],
    colour: "#b8a12b",
  },
  {
    words: ["Albert Rivera", "Albert Rivera Díaz"],
    colour: "var(--cs)",
  },
  {
    words: ["Alfons López Tena"],
    colour: "var(--si)",
  },
  {
    words: ["Josep Antoni Duran i Lleida", "Duran i Lleida"],
    colour: "var(--unio)",
  },
  {
    words: ["Pere Navarro", "Pere Navarro i Morera"],
    colour: "var(--psc)",
  },
  {
    words: ["Alícia Sánchez-Camacho"],
    colour: "var(--ppc)",
  },
  {
    words: ["Joan Herrera", "Joan Herrera i Torres"],
    colour: "var(--icv)",
  },
  {
    words: ["Mariano Rajoy", "Rajoy"],
    colour: "var(--pp)",
  },
  {
    words: ["Soraya Sáenz de Santamaría", "soraya saenz de santamaria"],
    colour: "var(--pp)",
  },
  {
    words: ["Pedro Sánchez"],
    colour: "var(--psoe)",
  },
  {
    words: ["Eduardo Madina"],
    colour: "var(--psoe)",
  },
  {
    words: ["Susana Díaz", "Susana Diaz"],
    colour: "var(--psoe)",
  },
  {
    words: ["Felipe González"],
    colour: "var(--psoe)",
  },
  {
    words: ["Alfredo Pérez Rubalcaba", "Rubalcaba"],
    colour: "var(--psoe)",
  },
  {
    words: ["Joan Laporta i Estruch", "Joan Laporta"],
    colour: "var(--si)",
  },
  {
    words: ["Ada Colau"],
    colour: "var(--bcomu)",
  },
  {
    words: ["Àngel Ros", "Angel Ros", "angel ros"],
    colour: "var(--psc)",
    transform: "Àngel Ros",
  },
  {
    words: ["Montserrat Tura", "montserrat tura"],
    colour: "var(--psc)",
    transform: "Montserrat Tura",
  },
  {
    words: ["Ernest Maragall", "ernest maragall"],
    colour: "var(--erc)",
    transform: "Ernest Maragall",
  },
  {
    words: ["Pasqual Maragall", "pasqual maragall"],
    colour: "var(--psc)",
    transform: "Pasqual Maragall",
  },
  {
    words: ["Núria Parlon", "Nuria Parlon", "nuria parlon"],
    colour: "var(--psc)",
    transform: "Núria Parlon",
  },
  {
    words: ["Miquel Iceta", "miquel iceta"],
    colour: "var(--psc)",
    transform: "Miquel Iceta",
  },
  {
    words: ["Gabriel Rufian", "gabriel rufian", "Grabriel Rufián"],
    colour: "var(--erc)",
    transform: "Gabriel Rufián",
  },
];
