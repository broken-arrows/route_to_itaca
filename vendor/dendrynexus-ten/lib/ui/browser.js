/* dendry
 * http://github.com/idmillington/dendry
 *
 * MIT License
 */
/*jshint indent:2 */
(function ($) {
  "use strict";

  var contentToHTML = require("./content/html");
  var engine = require("../engine");
  var persistence = require("../persistence");
  var saveLabel = require("./save-label");

  var canonicalSlot = function (slot) {
    if (slot === "a0") {
      return "auto-1";
    }
    if (slot === "a1") {
      return "auto-2";
    }
    return "manual-" + (Number(slot) + 1);
  };

  var BrowserUserInterface = function (game, $content) {
    this.game = game;
    this.$content = $content;
    this._registerEvents();

    this.dendryEngine = new engine.DendryEngine(this, game);
    // TODO: refactor how the settings work - move it all within a single object
    this.base_settings = {
      disable_bg: false,
      animate: false,
      animate_bg: true,
      disable_audio: false,
      show_portraits: true,
      dark_mode: false,
    };
    // current settings...
    this.current_settings = {
      disable_bg: false,
      animate: false,
      animate_bg: true,
      disable_audio: false,
      show_portraits: true,
      dark_mode: false,
    };
    this.disable_bg = false;
    this.animate = false;
    this.animate_bg = true;
    this.disable_audio = false;
    // backgrounds and portraits are 100% optional, and most games will not use them.
    this.show_portraits = true;
    this.fade_time = 600;
    this.bg_fade_out_time = 200;
    this.bg_fade_in_time = 1000;
    this.sound_fade_time = 2000;
    this.contentToHTML = contentToHTML;

    // sprites
    this.spriteLocs = {
      topLeft: 1,
      topRight: 1,
      bottomLeft: 1,
      bottomRight: 1,
    };
    // current HTMLAudioElement
    this.currentAudio = null;
    // current audio url
    this.currentAudioURL = "";
    this.audioQueue = [];
    // playlist is used for shuffling...
    this.audioPlaylist = [];
    // flag for determining if we're on a new page, up until the first choice.
    this.onNewPage = false;

    // for saving
    this._configurePersistence();
    this.max_slots = 8; // max save slots
    this.DateOptions = {
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
      year: "numeric",
      month: "short",
      day: "numeric",
    };
  };
  engine.UserInterface.makeParentOf(BrowserUserInterface);

  BrowserUserInterface.prototype._configurePersistence = function () {
    var info = this.game.info || this.game;
    this.storage_id = info.storageId || this.game.storageId;
    this.game_version = info.version || this.game.version;
    this.settings_key = this.storage_id + ":settings-old";
    this.saveStore = persistence.createSaveStore({
      storage: localStorage,
      storageId: this.storage_id,
      gameVersion: this.game_version,
    });
  };

  // ------------------------------------------------------------------------
  // Main API

  //load a game as a json file from a url, and then run the game...
  BrowserUserInterface.prototype.loadGame = function (url) {
    var that = this;
    if (!url.endsWith(".json")) {
      if (url.endsWith("/")) {
        url = url + "game.json";
      } else {
        url = url + "/game.json";
      }
    }
    fetch(url)
      .then((response) => response.text())
      .then((json) => {
        game = engine.convertJSONToGame(json, function (err, game) {
          if (err) {
            throw err;
          }
          return game;
        });
        that.game = game;
        that.dendryEngine = new engine.DendryEngine(that, game);
        that._configurePersistence();
        that.dendryEngine.beginGame();
      })
      .catch((err) => console.log(err));
  };

  BrowserUserInterface.prototype.displayContent = function (
    paragraphs,
    faceImage,
  ) {
    var $html = $(contentToHTML.convert(paragraphs));
    // TODO: maybe face image visibility should be controlled by a different setting?
    var hasImage = false;
    if (faceImage && this.show_portraits) {
      hasImage = true;
      // convert faceImage into an html object
      console.log(faceImage);
      //var cardEl = $('<div>').addClass('face-figure');
      var cardEl = document.createElement("div");
      cardEl.className = "face-figure";
      //var $image = $('<img>').addClass('face-img').attr({src : faceImage});
      var image = new Image();
      image.className = "face-img";
      cardEl.appendChild(image);
      $html.splice(1, 0, cardEl);
      image.src = faceImage;
      /*
        if (!this.animate) {
            var that = this;
            image.onload = function() {
                that.$content.append($html);
                console.log('image loaded');
            };
            image.src = faceImage;
        }
        */
    }
    if (this.animate) {
      $html.fadeIn(this.fade_time);
      this.$content.append($html);
    } else {
      if (!hasImage) {
        this.$content.append($html);
      } else {
        this.$content.append($html);
      }
    }
    $html.focus();
    // allow user to add custom stuff on display content (for sidebar in this case)
    if (window && window.onDisplayContent) {
      window.onDisplayContent();
    }
  };

  BrowserUserInterface.prototype.displayGameOver = function () {
    var $p = $("<p>").text(this.getGameOverMsg()).addClass("game-over");
    if (this.animate) {
      $p.fadeIn(this.fade_time);
      this.$content.append($p);
    } else {
      this.$content.append($p);
    }
    $p.focus();
  };

  BrowserUserInterface.prototype.displayChoices = function (choices) {
    var $ul = $("<ul>").addClass("choices");
    for (var i = 0; i < choices.length; ++i) {
      var choice = choices[i];

      var title = contentToHTML.convertLine(choice.title);
      var subtitle = "";
      if (choice.subtitle !== undefined) {
        subtitle = contentToHTML.convertLine(choice.subtitle);
      }

      var $li = $("<li>");
      var $titleHolder = $li;
      if (choice.canChoose) {
        $titleHolder = $("<a>").attr({ href: "#", "data-choice": i });
        $li.html($titleHolder);
      } else {
        $titleHolder.addClass("unavailable");
      }
      $titleHolder.html(title);
      if (
        choice.checkQuality &&
        choice.difficulty &&
        choice.successProb !== undefined
      ) {
        if (subtitle) {
          subtitle += "<br>";
        }
        subtitle += "Check: " + choice.checkQuality + "<br>";
        subtitle +=
          "Difficulty: " +
          choice.difficulty +
          " (" +
          Math.floor(choice.successProb * 100) +
          "%)";
      }
      if (subtitle) {
        $li.append($("<div>").addClass("subtitle").html(subtitle));
      }

      $ul.append($li);
    }
    if (this.animate) {
      $ul.fadeIn(this.fade_time);
      this.$content.append($ul);
    } else {
      this.$content.append($ul);
    }
    $ul.focus();
    if (this.onNewPage) {
      this.onNewPage = false;
      if (window && window.onNewPage) {
        window.onNewPage();
      }
    }
  };

  BrowserUserInterface.prototype.newPage = function () {
    if (this.animate) {
      var $content = this.$content;
      this.$content.empty();
      this.$content.children().fadeOut(this.fade_time, function () {});
    } else {
      this.$content.empty();
    }
    this.onNewPage = true;
  };

  BrowserUserInterface.prototype.setStyle = function (style) {
    this.$content.removeClass();
    if (style !== undefined) {
      this.$content.addClass(style);
    }
  };

  BrowserUserInterface.prototype.removeChoices = function () {
    $(".choices", this.$content).remove();
    $(".hidden", this.$content).remove();
  };

  BrowserUserInterface.prototype.beginOutput = function () {
    $("#read-marker", this.$content).remove();
    this.$content.append($("<hr>").attr("id", "read-marker"));
  };

  BrowserUserInterface.prototype.endOutput = function () {
    var $marker = $("#read-marker");
    if (this.animate) {
      if ($marker.length > 0) {
        $("html, body").animate(
          { scrollTop: $marker.offset().top },
          this.fade_time,
        );
      } else {
        $("html, body").animate({ scrollTop: 0 }, this.fade_time);
      }
    }
  };

  BrowserUserInterface.prototype.signal = function (data) {
    // TODO: implement signals - signals contain signal, event, and id
    console.log(data);
    var signal = data.signal;
    var event = data.event; // scene-arrival, scene-display, scene-departure, quality-change
    var scene_id = data.id;
    // TODO: handle this in the game.js for each specific game
    if (window && window.handleSignal) {
      window.handleSignal(signal, event, scene_id);
    }
  };

  // dendrynexus displays
  // displays the hand.
  BrowserUserInterface.prototype.displayHand = function (hand, maxCards) {
    if (window && window.displayHand) {
      window.displayHand(hand, maxCards);
      return null;
    }
    var handDescription = "Hand - click a card to play.";
    if (window.handDescription) {
      handDescription = window.handDescription;
    }
    if (this.dendryEngine.state.qualities.handDescription) {
      handDescription = this.dendryEngine.state.qualities.handDescription;
    }
    var $handEl = $(".hand");
    var hasOldHand = false;
    if ($handEl.length == 0) {
      $handEl = $("<ul>").addClass("hand");
      this.$content.append($("<hr>"));
      this.$content.append(
        $("<p>").addClass("hand-description").text(handDescription),
      );
    } else {
      $handEl.empty();
      hasOldHand = true;
    }
    // display the hand
    for (var i = 0; i < maxCards; i++) {
      var $cardEl = $("<li>").addClass("card-in-hand");
      if (hand[i]) {
        var card = hand[i];
        // create an <a> element, with an image nested inside.
        var $cardLink = $("<a>")
          .addClass("card")
          .attr({ href: "#", "card-id": card.id, title: card.title });
        var $title = $("<span>").addClass("card-caption").text(card.title);
        // if there is an image, set the image; otherwise, set image to a gradient?
        if (card.image) {
          var $cardImage = $("<img>")
            .addClass("card-img")
            .attr({ src: card.image });
          $cardLink.append($cardImage);
        } else {
        }
        if (card.subtitle) {
          var $cardSubtitle = $("<span>")
            .addClass("card-tooltip")
            .text(card.subtitle);
          $cardLink.append($cardSubtitle);
        }
        $cardEl.append($cardLink);
        $cardEl.append($title);
        $handEl.append($cardEl);
      } else {
        var $blankCardDiv = $("<div>").addClass("blank-card");
        $cardEl.append($blankCardDiv);
      }
      $handEl.append($cardEl);
    }
    if (!hasOldHand) {
      this.$content.append($handEl);
    }
  };

  BrowserUserInterface.prototype.displayDecks = function (decks) {
    if (window && window.displayDecks) {
      window.displayDecks(decks);
      return null;
    }
    var deckDescription = "Decks - click a deck to draw a card.";
    if (window.deckDescription) {
      deckDescription = window.deckDescription;
    }
    if (this.dendryEngine.state.qualities.deckDescription) {
      deckDescription = this.dendryEngine.state.qualities.deckDescription;
    }
    this.$content.append($("<hr>"));
    this.$content.append(
      $("<p>").addClass("deck-description").text(deckDescription),
    );
    var $decksEl = $("<ul>").addClass("decks");
    for (var deck of decks) {
      var $deckEl = $("<li>").addClass("deck");
      // create an <a> element, with an image nested inside.
      var $deckLink = $("<a>")
        .addClass("card")
        .attr({ href: "#", "card-id": deck.id, title: deck.title });
      var $title = $("<span>").addClass("card-caption").text(deck.title);
      // if there is an image, set the image; otherwise, set image to a gradient?
      if (deck.image) {
        var $deckImage = $("<img>")
          .addClass("card-img")
          .attr({ src: deck.image });
        $deckLink.append($deckImage);
      } else {
        // TODO: set alternative background for $deckLink to a gradient
      }
      if (deck.subtitle) {
        // if there's a subtitle, create a tooltip
        var $deckSubtitle = $("<span>")
          .addClass("card-tooltip")
          .text(deck.subtitle);
        $deckLink.append($deckSubtitle);
      }
      if (!deck.canChoose) {
        $deckEl = $deckEl.addClass("unavailable-card");
      }
      $deckEl.append($deckLink);
      $deckEl.append($title);
      $decksEl.append($deckEl);
    }
    this.$content.append($decksEl);
  };

  // displays pinned cards for dendrynexus
  BrowserUserInterface.prototype.displayPinnedCards = function (cards) {
    if (cards.length == 0) {
      return null;
    }
    if (window && window.displayPinnedCards) {
      window.displayPinnedCards(cards);
      return null;
    }
    var pinnedCardsDescription = "Pinned cards - click a card to play.";
    var pinnedParlamentDescription = "";
    if (window.pinnedCardsDescription) {
      pinnedCardsDescription = window.pinnedCardsDescription;
    }
    if (this.dendryEngine.state.qualities.pinnedCardsDescription) {
      pinnedCardsDescription =
        this.dendryEngine.state.qualities.pinnedCardsDescription;
    }
    this.$content.append($("<hr>"));
    this.$content.append(
      $("<p>").addClass("pinned-text-description").text(pinnedCardsDescription),
    );
    if (this.dendryEngine.state.qualities.pinnedParlamentDescription) {
      pinnedParlamentDescription =
        this.dendryEngine.state.qualities.pinnedParlamentDescription;
      this.$content.append(
        $("<p>")
          .addClass("pinned-text-description")
          .text(pinnedParlamentDescription),
      );
    }
    var $cardsEl = $("<ul>").addClass("pinned-cards");
    for (var card of cards) {
      var $cardEl = $("<li>").addClass("pinned-card");
      // create an <a> element, with an image nested inside.
      var $cardLink = $("<a>")
        .addClass("card")
        .attr({ href: "#", "card-id": card.id, title: card.title });
      var $title = $("<span>").addClass("card-caption").text(card.title);
      // if there is an image, set the image; otherwise, set image to a gradient?
      if (card.image) {
        var $cardImage = $("<img>")
          .addClass("card-img")
          .attr({ src: card.image });
        $cardLink.append($cardImage);
      } else {
      }
      if (card.subtitle) {
        var $cardSubtitle = $("<span>")
          .addClass("card-tooltip")
          .text(card.subtitle);
        $cardLink.append($cardSubtitle);
      }
      $cardEl.append($cardLink);
      $cardEl.append($title);
      $cardsEl.append($cardEl);
    }
    this.$content.append($cardsEl);
  };

  // visual extensions

  BrowserUserInterface.prototype.setBg = function (image_url) {
    if (this.disable_bg) {
      $("#bg1").addClass("content_hidden");
      $("#bg1").removeClass("content_visible");
      $("#bg1").css("background-image", "none");
    } else if (!image_url || image_url == "none" || image_url == "null") {
      if (this.animate_bg) {
        $("#bg1").addClass("content_hidden");
        $("#bg1").removeClass("content_visible");
        setTimeout(function () {
          $("#bg1").css("background-image", "none");
          $("#bg1").removeClass("content_hidden");
          $("#bg1").addClass("content_visible");
        }, 100);
      } else {
        $("#bg1").css("background-image", "none");
      }
    } else if (
      image_url.startsWith("#") ||
      image_url.startsWith("rgba(") ||
      image_url.startsWith("rgb(")
    ) {
      if (this.animate_bg) {
        $("#bg1").fadeOut(this.bg_fade_out_time, function () {
          $("#bg1").css("background-image", "none");
          $("#bg1").css("background-color", image_url);
        });
        $("#bg1").fadeIn(this.bg_fade_in_time, function () {
          $("#bg2").css("background-image", "none");
        });
        console.log("changing background color " + image_url);
      } else {
        $("#bg1").css("background-image", "none");
        $("#bg1").css("bacground-color", image_url);
      }
    } else if (image_url.startsWith("linear-gradient(")) {
      if (this.animate_bg) {
        $("#bg1").fadeOut(this.bg_fade_out_time, function () {
          $("#bg1").css("background-image", image_url);
        });
        $("#bg1").fadeIn(this.bg_fade_in_time, function () {
          $("#bg2").css("background-image", image_url);
        });
        console.log("changing background gradient " + image_url);
      } else {
        $("#bg1").css("background-image", image_url);
      }
    } else {
      if (this.animate_bg) {
        $("#bg1").fadeOut(this.bg_fade_out_time, function () {
          $("#bg1").css("background-image", 'url("' + image_url + '")');
        });
        $("#bg1").fadeIn(this.bg_fade_in_time, function () {
          $("#bg2").css("background-image", $("#bg1").css("background-image"));
        });
      } else {
        $("#bg1").css("background-image", 'url("' + image_url + '")');
      }
    }
  };

  // set sprites given data
  // data is a list of two-element lists, where the first element is location
  // (one of topLeft, topRight, bottomLeft, bottomRight)
  // and the second element is the sprite.
  BrowserUserInterface.prototype.setSprites = function (data) {
    if (window && window.setSprites) {
      window.setSprites(data);
      return;
    }
    if (!this.show_portraits || data == "none" || data == "clear") {
      $("#topLeftSprite")
        .children()
        .fadeOut(this.fade_time, function () {
          $("#topLeftSprite").empty();
        });
      $("#topRightSprite")
        .children()
        .fadeOut(this.fade_time, function () {
          $("#topRightSprite").empty();
        });
      $("#bottomLeftSprite")
        .children()
        .fadeOut(this.fade_time, function () {
          $("#bottomLeftSprite").empty();
        });
      $("#bottomRightSprite")
        .children()
        .fadeOut(this.fade_time, function () {
          $("#bottomRightSprite").empty();
        });
      return;
    } else {
      if (data instanceof Array) {
        for (var i = 0; i < data.length; i++) {
          var loc = data[i][0];
          var img = data[i][1];
          this.setSprite(loc, img);
        }
      } else if (data) {
        for (var key in Object.keys(data)) {
          sprites.push([key, data[key]]);
        }
      }
    }
  };

  BrowserUserInterface.prototype.setSprite = function (loc, img) {
    if (!this.show_portraits) {
      return;
    }
    if (window && window.setSprite) {
      window.setSprite(loc, img);
      return;
    }
    loc = loc.toLowerCase();
    var targetSprite;
    if (loc == "topleft") {
      targetSprite = $("#topLeftSprite");
    } else if (loc == "topright") {
      targetSprite = $("#topRightSprite");
    } else if (loc == "bottomleft") {
      targetSprite = $("#bottomLeftSprite");
    } else if (loc == "bottomright") {
      targetSprite = $("#bottomRightSprite");
    }
    //targetSprite.empty();
    if (img == "none" || img == "clear") {
      delete this.dendryEngine.state.sprites[loc];
      targetSprite.fadeOut(this.fade_time, function () {
        targetSprite.empty();
      });
      return;
    } else {
      this.dendryEngine.state.sprites[loc] = img;
      targetSprite.fadeOut(this.fade_time, function () {
        targetSprite.emtpy();
        var image = new Image();
        image.src = img;
        targetSprite.append(image);
        console.log("fadeIn");
        targetSprite.fadeIn(this.fade_time);
      });
    }
  };

  BrowserUserInterface.prototype.setSpriteStyle = function (loc, style) {
    if (window && window.setSpriteStyle) {
      window.setSpriteStyle(loc, style);
      return;
    }
    var targetSprite;
    if (loc == "topleft") {
      targetSprite = $("#topLeftSprite");
    } else if (loc == "topright") {
      targetSprite = $("#topRightSprite");
    } else if (loc == "bottomleft") {
      targetSprite = $("#bottomLeftSprite");
    } else if (loc == "bottomright") {
      targetSprite = $("#bottomRightSprite");
    } else {
      return;
    }
    targetSprite.css(style);
  };

  // play audio with js
  // audio is a space-separated string with at least one entry.
  // the first entry will be a file url.
  // the second-nth entries are words describing how the file will be played:
  // 'queue' for playing the music next after the current audio ends
  // 'loop' if this music will loop indefinitely.
  // 'nofade' if the sound will be played instantly without a fadein or fadeout.
  // TODO: have a list of audio files...
  BrowserUserInterface.prototype.audio = function (audio) {
    if (this.disable_audio) {
      if (this.currentAudio) {
        this.currentAudio.pause();
        this.currentAudio.loop = false;
      }
      return;
    }
    var audioData = audio.split(" ");
    var audioFiles = [];
    var isLoop = false;
    var isQueue = false;
    var noFade = false;
    var isShuffle = false;
    var isClear = false;
    for (var name of audioData) {
      if (name == "loop") {
        isLoop = true;
      } else if (name == "queue") {
        isQueue = true;
      } else if (name == "nofade") {
        noFade = true;
      } else if (name == "shuffle") {
        isShuffle = true;
      } else if (name == "clear") {
        isClear = true;
      } else {
        audioFiles.push(name);
      }
    }
    if (isClear) {
      this.audioPlaylist = [];
    }
    if (audioFiles.length >= 1 || isShuffle) {
      this.audioPlaylist = this.audioPlaylist.concat(audioFiles);
    }
    var audioFile = audioFiles[0];
    var currentAudio = this.currentAudio;
    var fadeTime = this.sound_fade_time;
    var loopCurrent = false;
    var playlist = this.audioPlaylist;
    // stop playing
    if (audioFile == "null" || audioFile == "none") {
      if (this.currentAudio) {
        $(currentAudio).animate(
          { volume: 0 },
          this.sound_fade_time,
          function () {
            currentAudio.pause();
          },
        );
        this.currentAudio.loop = false;
      }
    } else {
      // fadeout current audio, then fade-in new audio
      console.log(
        "new audio:",
        audioFile,
        "current audio:",
        this.currentAudioURL,
      );
      if (
        this.currentAudio &&
        (this.currentAudioURL == audioFile || isQueue || isShuffle)
      ) {
        if (!currentAudio.ended && !currentAudio.paused) {
          console.log("adding music to queue");
          this.audioQueue.push(audioFile);
          var audioQueue = this.audioQueue;
          this.currentAudio.onended = function () {
            var newAudio;
            if (isQueue) {
              newAudio = audioQueue.pop();
              console.log("playing from queue");
            } else if (isShuffle) {
              var index = Math.floor(Math.random() * playlist.length);
              newAudio = playlist[index];
              console.log("playing from playlist");
            }
            if (newAudio) {
              currentAudio.src = newAudio;
              console.log("Now playing", newAudio);
              currentAudio.play();
              $(currentAudio).animate({ volume: 1 }, fadeTime);
              window.dendryUI.currentAudioURL = newAudio;
            }
          };
        } else {
          this.currentAudioURL = audioFile;
          currentAudio.src = audioFile;
          console.log("Fading in new audio");
          currentAudio.volume = 0;
          currentAudio.play();
          $(currentAudio).animate({ volume: 1 }, fadeTime);
          this.currentAudio.onended = function () {
            var newAudio;
            if (isQueue) {
              newAudio = audioQueue.pop();
              console.log("playing from queue");
            } else if (isShuffle) {
              var index = Math.floor(Math.random() * playlist.length);
              newAudio = playlist[index];
              console.log("playing from playlist");
            }
            if (newAudio) {
              currentAudio.src = newAudio;
              console.log("Now playing", newAudio);
              currentAudio.play();
              $(currentAudio).animate({ volume: 1 }, fadeTime);
              window.dendryUI.currentAudioURL = newAudio;
            }
          };
        }
      } else if (this.currentAudio) {
        // not queue or shuffle, so we stop playing the current audio.
        this.currentAudioURL = audioFile;
        console.log("currentAudio present,  fading out current audio");
        // reset the current audio function
        currentAudio.onended = function () {};
        if (noFade) {
          currentAudio.pause();
          currentAudio.src = audioFile;
          currentAudio.play();
        } else {
          $(currentAudio).animate(
            { volume: 0 },
            this.sound_fade_time,
            function () {
              console.log(currentAudio);
              currentAudio.src = audioFile;
              console.log("Fading in new audio");
              currentAudio.play();
              $(currentAudio).animate({ volume: 1 }, fadeTime);
            },
          );
        }
      } else if (!this.currentAudio) {
        this.currentAudio = new Audio(audioFile);
        this.currentAudio.volume = 0;
        this.currentAudio.play();
        $(this.currentAudio).animate({ volume: 1 }, this.sound_fade_time);
        currentAudio = this.currentAudio;
        if (isShuffle) {
          this.currentAudio.onended = function () {
            var index = Math.floor(Math.random() * playlist.length);
            var newAudio = playlist[index];
            if (newAudio) {
              currentAudio.src = newAudio;
              console.log("playing from shuffle");
              console.log("Now playing", newAudio);
              currentAudio.play();
              $(currentAudio).animate({ volume: 1 }, fadeTime);
              // asdkfl;;sajd;lkjafdsdsaf;kjldjsfa;kl
              window.dendryUI.currentAudioURL = newAudio;
            }
          };
        }
      }
      if (isLoop) {
        this.currentAudio.loop = true;
      } else {
        this.currentAudio.loop = false;
      }
      // https://stackoverflow.com/questions/7451508/html5-audio-playback-with-fade-in-and-fade-out
    }
  };

  BrowserUserInterface.prototype.saveSettings = function () {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(
        this.settings_key,
        JSON.stringify({
          animate: this.animate,
          disableBg: this.disable_bg,
          animateBg: this.animate_bg,
          showPortraits: this.show_portraits,
          disableAudio: this.disable_audio,
          darkMode: this.dark_mode,
        }),
      );
    }
  };

  // TODO: this could be much cleaner...
  BrowserUserInterface.prototype.loadSettings = function (defaultSettings) {
    var defaults = {
      animate: false,
      disable_bg: false,
      animate_bg: true,
      show_portraits: true,
      disable_audio: false,
      dark_mode: false,
    };
    var storedNames = {
      animate: "animate",
      disable_bg: "disableBg",
      animate_bg: "animateBg",
      show_portraits: "showPortraits",
      disable_audio: "disableAudio",
      dark_mode: "darkMode",
    };
    var stored = null;
    if (typeof localStorage !== "undefined") {
      try {
        stored = JSON.parse(localStorage.getItem(this.settings_key));
      } catch (error) {
        stored = null;
      }
      for (var prop in defaults) {
        if (defaults.hasOwnProperty(prop)) {
          var storedName = storedNames[prop];
          if (stored && stored.hasOwnProperty(storedName)) {
            this[prop] = stored[storedName];
          } else {
            if (defaultSettings && defaultSettings.hasOwnProperty(prop)) {
              this[prop] = defaultSettings[prop];
            } else {
              this[prop] = defaults[prop];
            }
          }
        }
      }
    }
  };

  BrowserUserInterface.prototype.toggle_audio = function (enable_audio) {
    if (enable_audio) {
      this.disable_audio = false;
      if (this.currentAudio) {
        this.currentAudio.play();
      }
    } else {
      if (this.currentAudio) {
        this.currentAudio.pause();
        this.currentAudio.loop = false;
      }
      this.disable_audio = true;
    }
  };

  // save functions
  BrowserUserInterface.prototype._saveMetadata = function () {
    return { sceneId: this.dendryEngine.state.sceneId };
  };

  BrowserUserInterface.prototype._displaySaveError = function (action, result) {
    var detail = result && result.error && result.error.message;
    window.alert(action + " failed." + (detail ? "\n" + detail : ""));
  };

  BrowserUserInterface.prototype._loadSaveResult = function (result) {
    if (result.status === "missing") {
      window.alert("No save available.");
      return false;
    }
    if (result.status === "unsupported") {
      window.alert(
        "This save uses an unsupported save format and cannot be loaded.",
      );
      return false;
    }
    if (result.status === "corrupt" || result.status === "unreadable") {
      window.alert("This save is corrupt or unreadable and cannot be loaded.");
      return false;
    }
    if (
      result.compatibility !== "compatible" &&
      !window.confirm(
        "This save was created with a different or unknown game version. Load it anyway?",
      )
    ) {
      return false;
    }
    this.dendryEngine.setState(result.record.state);
    this.hideSaveSlots();
    window.alert("Loaded.");
    return true;
  };

  BrowserUserInterface.prototype.autosave = function () {
    var previous = this.saveStore.export("auto-1");
    if (previous.ok) {
      // Import preserves the complete previous envelope, including savedAt.
      var rotated = this.saveStore.import("auto-2", previous.data);
      if (!rotated.ok) {
        this._displaySaveError("Autosave rotation", rotated);
        return;
      }
    } else if (previous.error.code !== "missing-save") {
      this._displaySaveError("Autosave rotation", previous);
      return;
    }
    var result = this.saveStore.write(
      "auto-1",
      this.dendryEngine.getExportableState(),
      this._saveMetadata(),
    );
    if (!result.ok) {
      this._displaySaveError("Autosave", result);
    }
    this.populateSaveSlots(this.max_slots, 2);
  };

  BrowserUserInterface.prototype.saveSlot = function (slot) {
    var result = this.saveStore.write(
      canonicalSlot(slot),
      this.dendryEngine.getExportableState(),
      this._saveMetadata(),
    );
    if (!result.ok) {
      this._displaySaveError("Save", result);
    }
    this.populateSaveSlots(this.max_slots, 2);
  };

  BrowserUserInterface.prototype.loadSlot = function (slot) {
    this._loadSaveResult(this.saveStore.read(canonicalSlot(slot)));
  };

  BrowserUserInterface.prototype.deleteSlot = function (slot) {
    var result = this.saveStore.remove(canonicalSlot(slot));
    if (!result.ok) {
      this._displaySaveError("Delete", result);
    } else if (!result.existed) {
      window.alert("No save available.");
    }
    this.populateSaveSlots(this.max_slots, 2);
  };

  BrowserUserInterface.prototype.exportSlot = function (slot) {
    var result = this.saveStore.export(canonicalSlot(slot));
    if (result.ok) {
      var a = document.createElement("a");
      var file = new Blob([result.data], { type: "application/json" });
      a.href = URL.createObjectURL(file);
      a.download = "save.txt";
      a.click();
    } else {
      this._displaySaveError("Export", result);
    }
  };

  BrowserUserInterface.prototype.importSave = function (doc_id) {
    var that = this;
    function onFileLoad(e) {
      var data = e.target.result;
      var slot = null;
      for (var i = 1; i <= that.max_slots; i++) {
        if (that.saveStore.read("manual-" + i).status === "missing") {
          slot = "manual-" + i;
          break;
        }
      }
      if (!slot) {
        window.alert(
          "All save slots are occupied. Delete a save before importing.",
        );
        return;
      }
      var imported = that.saveStore.import(slot, data);
      if (!imported.ok) {
        that._displaySaveError("Import", imported);
        return;
      }
      that.populateSaveSlots(that.max_slots, 2);
      that._loadSaveResult(that.saveStore.read(slot));
    }
    var uploader = document.getElementById(doc_id);
    var reader = new FileReader();
    var file = uploader.files[0];
    if (!file) {
      window.alert("Choose a save file to import.");
      return;
    }
    reader.onload = onFileLoad;
    reader.readAsText(file);
  };

  BrowserUserInterface.prototype.populateSaveSlots = function (
    max_slots,
    max_auto_slots,
  ) {
    // this fills in the save information
    var that = this;
    function createLoadListener(i) {
      return function (evt) {
        that.loadSlot(i);
      };
    }
    function createSaveListener(i) {
      return function (evt) {
        that.saveSlot(i);
      };
    }
    function createDeleteListener(i) {
      return function (evt) {
        that.deleteSlot(i);
      };
    }
    function createExportListener(i) {
      return function (evt) {
        that.exportSlot(i);
      };
    }
    function populateSlot(id) {
      var entry = that.saveStore.read(canonicalSlot(id));
      var save_element = document.getElementById("save_info_" + id);
      var save_button = document.getElementById("save_button_" + id);
      var delete_button = document.getElementById("delete_button_" + id);
      var export_button = document.getElementById("export_button_" + id);
      save_button.onclick = null;
      delete_button.onclick = null;
      export_button.onclick = null;
      save_button.disabled = false;
      delete_button.disabled = false;
      export_button.disabled = false;
      if (entry.status === "ready") {
        var meta = entry.record.meta || {};
        var savedAt = new Date(meta.savedAt);
        var timestamp = isNaN(savedAt.getTime())
          ? String(meta.savedAt || "")
          : savedAt.toLocaleString(undefined, that.DateOptions);
        var state = entry.record.state;
        var label =
          String(meta.sceneId || (state && state.sceneId) || "Unknown scene") +
          "\n(" +
          timestamp +
          ")";
        save_element.textContent = saveLabel.formatSaveTimestamp(label);
        if (entry.compatibility !== "compatible") {
          save_element.textContent += "\n(Game version warning)";
        }
        save_button.textContent = "Load";
        save_button.onclick = createLoadListener(id);
        delete_button.onclick = createDeleteListener(id);
        export_button.onclick = createExportListener(id);
      } else if (entry.status === "missing") {
        save_button.textContent = "Save";
        save_element.textContent = "Empty";
        save_button.onclick = createSaveListener(id);
        delete_button.disabled = true;
        export_button.disabled = true;
      } else {
        save_button.textContent = "Load";
        save_button.disabled = true;
        save_element.textContent =
          entry.status === "unsupported"
            ? "Unsupported save format"
            : entry.status === "corrupt"
              ? "Corrupt save"
              : "Unreadable save";
        delete_button.onclick = createDeleteListener(id);
        if (entry.status === "unreadable") {
          export_button.disabled = true;
        } else {
          export_button.onclick = createExportListener(id);
        }
      }
    }
    for (var i = 0; i < max_slots; i++) {
      populateSlot(i);
    }
    for (i = 0; i < max_auto_slots; i++) {
      populateSlot("a" + i);
    }
  };

  BrowserUserInterface.prototype.showSaveSlots = function () {
    if (this.dendryEngine.state.disableSaves) {
      window.alert("Saving and loading is currently disabled.");
      return;
    }
    var save_element = document.getElementById("save");
    save_element.style.display = "block";
    this.populateSaveSlots(this.max_slots, 2);
    var that = this;
    if (!save_element.onclick) {
      save_element.onclick = function (evt) {
        var target = evt.target;
        var save_element = document.getElementById("save");
        if (target == save_element) {
          that.hideSaveSlots();
        }
      };
    }
  };

  BrowserUserInterface.prototype.hideSaveSlots = function () {
    var save_element = document.getElementById("save");
    save_element.style.display = "none";
  };

  // functions for dealing with options
  BrowserUserInterface.prototype.setOption = function (option, toggle) {
    this[option] = toggle;
    this.saveSettings();
  };

  BrowserUserInterface.prototype.populateOptions = function () {
    var disable_bg = this.disable_bg;
    var animate = this.animate;
    var animate_bg = this.animate_bg;
    if (disable_bg) {
      $("#backgrounds_no")[0].checked = true;
    } else {
      $("#backgrounds_yes")[0].checked = true;
    }
    if (animate) {
      $("#animate_yes")[0].checked = true;
    } else {
      $("#animate_no")[0].checked = true;
    }
    if (animate_bg) {
      $("#animate_bg_yes")[0].checked = true;
    } else {
      $("#animate_bg_no")[0].checked = true;
    }
  };

  BrowserUserInterface.prototype.showOptions = function () {
    var save_element = document.getElementById("options");
    this.populateOptions();
    save_element.style.display = "block";
    if (!save_element.onclick) {
      save_element.onclick = function (evt) {
        var target = evt.target;
        var save_element = document.getElementById("options");
        if (target == save_element) {
          this.hideOptions();
        }
      };
    }
  };

  // ------------------------------------------------------------------------
  // Additional methods

  BrowserUserInterface.prototype.getGameOverMsg = function () {
    return "Game Over (reload to read again)";
  };

  BrowserUserInterface.prototype._registerEvents = function () {
    var that = this;
    this.$content.on("click", "ul.choices li a", function (event) {
      event.preventDefault();
      event.stopPropagation();
      var choice = parseInt($(this).attr("data-choice"));
      that.dendryEngine.choose(choice);
      return false;
    });
    this.$content.on("click", "ul.choices li", function (event) {
      event.preventDefault();
      event.stopPropagation();
      $("a", this).click();
      return false;
    });
    // dendrynexus - onclick for decks and cards
    this.$content.on("click", "ul.decks li a", function (event) {
      event.preventDefault();
      event.stopPropagation();
      var choice = $(this).attr("card-id");
      that.dendryEngine.drawCard(choice);
      return false;
    });
    this.$content.on("click", "ul.hand li a", function (event) {
      event.preventDefault();
      event.stopPropagation();
      var choice = $(this).attr("card-id");
      that.dendryEngine.playCard(choice);
      return false;
    });
    this.$content.on("click", "ul.pinned-cards li a", function (event) {
      event.preventDefault();
      event.stopPropagation();
      var choice = $(this).attr("card-id");
      that.dendryEngine.playPinnedCard(choice);
      return false;
    });
  };

  // ------------------------------------------------------------------------
  // Run when loaded.

  var main = function () {
    engine.convertJSONToGame(window.game.compiled, function (err, game) {
      if (err) {
        throw err;
      }

      var ui = new BrowserUserInterface(game, $("#content"));
      window.dendryUI = ui;
      // Allow the ui system to be customized before use.
      if (window.dendryModifyUI !== undefined) {
        // If it returns true, then we don't need to begin the game.
        var dontStart = window.dendryModifyUI(ui);
        if (dontStart) {
          return;
        }
      }
      ui.dendryEngine.beginGame();
    });
  };

  BrowserUserInterface.canonicalSlot = canonicalSlot;
  module.exports = BrowserUserInterface;

  if (typeof window !== "undefined" && window.game) {
    $(main);
  }
})(jQuery);
