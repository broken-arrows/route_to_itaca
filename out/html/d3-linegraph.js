/*
 * generating a line graph of multipile parties with multiple dates...
 * */

function addMonths(date, months) {
  date = new Date(date);
  var d = date.getDate();
  date.setMonth(date.getMonth() + months);
  if (date.getDate() != d) {
    date.setDate(0);
  }
  return date;
}

d3.linegraph = function (
  noTicks,
  noDots,
  parties,
  partyColors,
  partyNames,
  dataMax,
  dataMin,
  additionalMonths,
  options,
) {
  options = options || {};
  function validReading(value) {
    return Number.isFinite(value) && (options.variables || value !== 0);
  }
  /* params */
  if (!parties) {
    console.error("No parties defined for line graph!");
    return;
  }
  if (!partyColors) {
    console.error("No partyColors defined for line graph!");
    return;
  }
  if (!partyNames) {
    console.error("No partyNames defined for line graph!");
    return;
  }
  if (!additionalMonths || additionalMonths == null) {
    additionalMonths = 10;
  }

  // Declare the chart dimensions and margins.
  var width = 500;
  var height = 400;
  var marginTop = 32;
  var marginRight = 20;
  var marginBottom = 50;
  var marginLeft = 40;
  var events = [];
  function formatDate(date) {
    if (!window.ClassicI18n || window.ClassicI18n.locale === "en")
      return d3.utcFormat("%b. %Y")(date);
    return new Intl.DateTimeFormat(
      window.ClassicI18n ? window.ClassicI18n.locale : "en",
      { month: "short", year: "numeric", timeZone: "UTC" },
    ).format(date);
  }

  function linegraph(dataset) {
    dataset.each(function (data) {
      marginRight = options.variables ? 100 : 20;
      const dates = data.map((d) => new Date(d.date));
      // Map the data to an array of arrays of {x, y} tuples.
      const series = parties.map((party) =>
        data
          .map((d) =>
            validReading(d[party])
              ? { x: new Date(d.date), y: d[party], series: party }
              : null,
          )
          .filter((d) => d !== null && d.y !== null && d.y !== undefined),
      );

      // Declare the x (horizontal position) scale.
      const maxDate = d3.max([
        d3.max(dates),
        d3.max(events, (event) => new Date(event.date)),
      ]);
      var eventLines = events.map(function (event) {
        return event.labels || [event.label];
      });
      var startTime = options.variables ? +d3.min(dates) : +new Date(2012, 7);
      var endDate = addMonths(maxDate, additionalMonths);
      if (+endDate === startTime) {
        startTime -= 86400000;
        endDate = new Date(+endDate + 86400000);
      }
      // Leave room for labels left of early events without crossing the percentage axis.
      events.forEach(function (event, index) {
        var padding = Math.min(
          0.75,
          (eventLines[index].length * 14 + 8) /
            (width - marginLeft - marginRight),
        );
        startTime = Math.min(
          startTime,
          (+new Date(event.date) - padding * +endDate) / (1 - padding),
        );
      });
      const xScale = d3.scaleUtc(
        [new Date(startTime), endDate],
        [marginLeft, width - marginRight],
      );
      var plotTop = marginTop;
      var annotations = events.map(function (event, index) {
        return {
          date: event.date,
          label: event.label,
          labelX:
            xScale(new Date(event.date)) -
            6 -
            (eventLines[index].length - 1) * 14,
          labelY: plotTop + 4,
          lines: eventLines[index],
        };
      });
      var plotBottom = height - marginBottom;

      var xaxis = d3
        .axisBottom()
        .tickFormat(formatDate)
        .tickValues(dates)
        .scale(xScale);
      if (noTicks) {
        xaxis = d3.axisBottom().tickFormat(formatDate).ticks(10).scale(xScale);
      }

      // Declare the y (vertical position) scale.
      if (options.variables) {
        var values = series.flat().map(function (point) {
          return point.y;
        });
        var min = Math.min(0, d3.min(values) || 0);
        var max = Math.max(0, d3.max(values) || 0);
        var padding = Math.max(1, (max - min) * 0.1);
        dataMin =
          options.yMin == null ? (min < 0 ? min - padding : 0) : options.yMin;
        dataMax = options.yMax == null ? max + padding : options.yMax;
      } else if (!dataMax) {
        dataMax =
          d3.max(parties, (party) =>
            d3.max(data, (d) => d[party] ?? -Infinity),
          ) + 10;
        dataMin = 0;
      }
      const yScale = d3.scaleLinear([dataMin, dataMax], [plotBottom, plotTop]);
      if (options.variables && options.yMin == null && options.yMax == null)
        yScale.nice(5);

      //Create the SVG container.
      //const svg = d3.create("svg")
      //    .attr("width", width)
      //    .attr("height", height);
      var svg = d3.select(this);
      if (events.length)
        svg.attr("viewBox", "0 0 " + width + " " + (plotBottom + marginBottom));
      svg
        .insert("rect", ":first-child")
        .attr("class", "history-hover-surface")
        .attr("x", marginLeft)
        .attr("y", plotTop)
        .attr("width", width - marginRight - marginLeft)
        .attr("height", plotBottom - plotTop)
        .attr("fill", "transparent");

      // Add the x-axis.
      svg
        .append("g")
        .attr("transform", `translate(0,${plotBottom})`)
        .call(xaxis)
        .call(function (axis) {
          axis.select(".domain").attr("class", "domain history-axis-baseline");
        })
        .selectAll("text")
        .attr("text-anchor", "end")
        .attr("dx", "-0.8em")
        .attr("dy", "0.1em")
        .attr("transform", "rotate(-30)");

      // Add the y-axis.
      svg
        .append("g")
        .attr("transform", `translate(${marginLeft},0)`)
        .call(
          d3
            .axisLeft(yScale)
            .tickSize(0)
            .tickFormat(function (value) {
              return (
                value +
                (options.variables
                  ? options.valueSuffix == null
                    ? "%"
                    : options.valueSuffix
                  : "")
              );
            }),
        );
      if (
        options.variables &&
        yScale.domain()[0] < 0 &&
        yScale.domain()[1] > 0
      ) {
        svg
          .append("line")
          .attr("class", "history-zero")
          .attr("x1", marginLeft)
          .attr("x2", width - marginRight)
          .attr("y1", yScale(0))
          .attr("y2", yScale(0))
          .attr("stroke", "#8a8273")
          .attr("stroke-dasharray", "3 4");
      }

      // Each event gets one vertical line of text beside its exact-date marker.
      annotations.forEach(function (event) {
        var date = new Date(event.date);
        if (+date < +xScale.domain()[0] || +date > +maxDate) return;
        var marker = svg.append("g").attr("class", "history-event");
        marker.append("title").text(event.label + " · " + formatDate(date));
        marker
          .append("line")
          .attr("class", "history-event-date")
          .attr("x1", xScale(date))
          .attr("x2", xScale(date))
          .attr("y1", plotTop)
          .attr("y2", plotBottom)
          .attr("stroke", "#b8b8b8")
          .attr("stroke-width", 1);
        var label = marker
          .append("text")
          .attr("x", event.labelX)
          .attr("y", event.labelY)
          .attr(
            "transform",
            "rotate(-90 " + event.labelX + " " + event.labelY + ")",
          )
          .attr("text-anchor", "end")
          .attr("fill", "#6b655a")
          .attr("font-size", "12px")
          .attr("paint-order", "stroke")
          .attr("stroke", "#f4f4f4")
          .attr("stroke-width", 3)
          .attr("stroke-linejoin", "round");
        event.lines.forEach(function (line, index) {
          label
            .append("tspan")
            .attr("x", event.labelX)
            .attr("dy", index ? 14 : 0)
            .text(line + " ");
        });
      });

      const partyLine = (party) =>
        d3
          .line()
          .defined((d) => validReading(d[party]))
          .x((d) => xScale(new Date(d.date)))
          .y((d) => yScale(d[party]));

      // draw the lines
      for (const party of parties) {
        const partySeries = series[parties.indexOf(party)];
        if (partySeries.length === 0) continue;
        svg
          .append("path")
          .attr("fill", "none")
          .attr("stroke", partyColors[party])
          .attr("stroke-width", 1.5)
          .attr("class", party + " " + "party-line")
          .attr("id", party + "-line")
          .attr("series", party)
          .attr("d", partyLine(party)(data))
          .on("mouseover", function (d) {
            svg.selectAll(".party-line").attr("stroke-width", 0.1);
            svg.selectAll(".party-node").attr("fill-opacity", 0.1);
            svg.selectAll(".party-label").attr("opacity", 0.1);
            svg.selectAll("." + party + "-node").attr("fill-opacity", 1);
            svg.selectAll("." + party + "-label").attr("opacity", 1);
            d3.select(this).attr("stroke-width", 5);
          })
          .on("mouseout", function (d) {
            svg.selectAll(".party-line").attr("stroke-width", 1.5);
            svg.selectAll(".party-node").attr("fill-opacity", 1);
            svg.selectAll(".party-label").attr("opacity", 1);
          });
      }

      // Keep the final node when intermediate dots are hidden.
      svg
        .selectAll(".series")
        .data(series)
        .enter()
        .append("g")
        .selectAll(".point")
        .data((d) => (noDots ? d.slice(-1) : d))
        .enter()
        .append("circle")
        .attr(
          "class",
          (d) => d.series + " " + d.series + "-node " + "party-node",
        )
        .attr("fill", (d) => partyColors[d.series])
        .attr("series", (d) => d.series)
        .attr("r", 4)
        .attr("cx", (d) => xScale(d.x))
        .attr("cy", (d) => yScale(d.y))
        .on("mouseover", function (d) {
          const node = d3.select(this);
          const series = node.attr("series");
          svg.selectAll(".party-line").attr("stroke-width", 0.1);
          svg.selectAll(".party-node").attr("fill-opacity", 0.1);
          svg.selectAll(".party-label").attr("opacity", 0.1);
          svg.selectAll("." + series + "-node").attr("fill-opacity", 1);
          svg.selectAll("#" + series + "-line").attr("stroke-width", 5);
          svg.selectAll("." + series + "-label").attr("opacity", 1);
        })
        .on("mouseout", function (d) {
          svg.selectAll(".party-line").attr("stroke-width", 1.5);
          svg.selectAll(".party-node").attr("fill-opacity", 1);
          svg.selectAll(".party-label").attr("opacity", 1);
        });

      // Variable names need more space than party abbreviations.
      var labelPositions = {};
      if (options.variables) {
        var previousY = plotBottom + 18;
        series
          .filter(function (s) {
            return s.length;
          })
          .sort(function (a, b) {
            return a[a.length - 1].y - b[b.length - 1].y;
          })
          .forEach(function (s) {
            var last = s[s.length - 1];
            previousY = Math.min(yScale(last.y) + 5, previousY - 18);
            labelPositions[last.series] = previousY;
          });
      }

      // draw right-hand labels
      svg
        .selectAll(".labels")
        .data(series.filter((s) => s.length > 0))
        .enter()
        .append("text")
        .text((s) => partyNames[s[0].series])
        .attr("series", (s) => s[0].series)
        .attr("font-size", "0.8em")
        .attr("class", (s) => s[0].series + "-label party-label")
        .attr("x", (s) => xScale(s[s.length - 1].x) + 15)
        .attr("y", (s) =>
          options.variables
            ? labelPositions[s[0].series]
            : yScale(s[s.length - 1].y) + 5,
        )
        .attr("fill", (s) =>
          options.variables ? partyColors[s[0].series] : null,
        )
        .on("mouseover", function (d) {
          const text = d3.select(this);
          const series = text.attr("series");
          svg.selectAll(".party-line").attr("stroke-width", 0.1);
          svg.selectAll(".party-node").attr("fill-opacity", 0.1);
          svg.selectAll(".party-label").attr("opacity", 0.1);
          svg.selectAll("." + series + "-node").attr("fill-opacity", 1);
          svg.selectAll("#" + series + "-line").attr("stroke-width", 5);
          svg.selectAll("." + series + "-label").attr("opacity", 1);
        })
        .on("mouseout", function (d) {
          svg.selectAll(".party-line").attr("stroke-width", 1.5);
          svg.selectAll(".party-node").attr("fill-opacity", 1);
          svg.selectAll(".party-label").attr("opacity", 1);
        });

      var guide = svg
        .append("g")
        .attr("class", "history-date-guide")
        .attr("pointer-events", "none")
        .style("display", "none");
      var guideLine = guide
        .append("line")
        .attr("y1", plotTop)
        .attr("y2", plotBottom)
        .attr("stroke", "#8a8273")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "3 4")
        .attr("opacity", 0.65);
      var guideText = guide
        .append("text")
        .attr("y", plotTop - 6)
        .attr("text-anchor", "middle")
        .attr("fill", "#6b655a")
        .attr("font-size", "12px");
      svg
        .on("mousemove.history-date", function (event) {
          if (
            event.target.closest &&
            event.target.closest(".party-line, .party-node, .party-label")
          ) {
            guide.style("display", "none");
            return;
          }
          var point = d3.pointer(event, this);
          if (
            point[0] < marginLeft ||
            point[0] > width - marginRight ||
            point[1] < plotTop ||
            point[1] > plotBottom
          ) {
            guide.style("display", "none");
            return;
          }
          var time = +xScale.invert(point[0]);
          var sample = data.reduce(function (best, row) {
            return !best ||
              Math.abs(+new Date(row.date) - time) <
                Math.abs(+new Date(best.date) - time)
              ? row
              : best;
          }, null);
          if (!sample) {
            guide.style("display", "none");
            return;
          }
          var sampleX = xScale(new Date(sample.date));
          var captions = [formatDate(new Date(sample.date))];
          var week = sample.week;
          if (Number.isInteger(week) && week >= 1 && week <= 4)
            captions.push("Week " + week);
          var captionWidth = Math.min(
            width - marginRight - marginLeft,
            d3.max(captions, function (line) {
              return line.length * 6.5;
            }),
          );
          var labelX = Math.max(
            marginLeft + captionWidth / 2,
            Math.min(width - marginRight - captionWidth / 2, sampleX),
          );
          guide.style("display", null);
          guideLine.attr("x1", sampleX).attr("x2", sampleX);
          guide
            .selectAll("circle.history-guide-point")
            .data(
              parties.filter(function (party) {
                return validReading(sample[party]);
              }),
            )
            .join("circle")
            .attr("class", "history-guide-point")
            .attr("data-party", function (party) {
              return party;
            })
            .attr("cx", sampleX)
            .attr("cy", function (party) {
              return yScale(sample[party]);
            })
            .attr("r", 3.5)
            .attr("fill", function (party) {
              return partyColors[party];
            })
            .attr("stroke", "#f4f4f4")
            .attr("stroke-width", 1.5);
          guideText
            .attr("x", labelX)
            .attr("y", plotTop - 6 - (captions.length - 1) * 14)
            .selectAll("tspan")
            .data(captions)
            .join("tspan")
            .attr("x", labelX)
            .attr("dy", function (line, index) {
              return index ? 14 : 0;
            })
            .text(function (line) {
              return line;
            });
        })
        .on("mouseleave.history-date", function () {
          guide.style("display", "none");
        });
    });
  }

  linegraph.width = function (value) {
    if (!arguments.length) return width;
    width = value;
    return linegraph;
  };

  linegraph.events = function (value) {
    if (!arguments.length) return events;
    events = Array.isArray(value) ? value : [];
    return linegraph;
  };

  linegraph.height = function (value) {
    if (!arguments.length) return height;
    height = value;
    return linegraph;
  };

  linegraph.parties = function (value) {
    if (!arguments.length) return parties;
    parties = value;
    return linegraph;
  };

  linegraph.partyNames = function (value) {
    if (!arguments.length) return partyNames;
    partyNames = value;
    return linegraph;
  };

  linegraph.partyColors = function (value) {
    if (!arguments.length) return partyColors;
    partyColors = value;
    return linegraph;
  };

  return linegraph;
};
