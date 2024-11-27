// scatterPie.js

function renderScatterPlotWithPieCharts(tsneData, containerId) {
    // Set up the SVG container for the scatter plot
    const svgWidth = 800, svgHeight = 800;
    const svg = d3.select(containerId)
                  .append("svg")
                  .attr("width", svgWidth)
                  .attr("height", svgHeight);
  
    const numCategories = tsneData[0].ratios.length;
  
    // Define the color scale for the pie chart
    const colorScale = d3.scaleOrdinal()
                         .domain(d3.range(numCategories))
                         .range(d3.schemeCategory10.concat(d3.schemeSet3).slice(0, numCategories));
  
    // Define scales for positioning the scatter plot
    const xScale = d3.scaleLinear()
                     .domain(d3.extent(tsneData, d => d.x))
                     .range([20, svgWidth - 20]);
    const yScale = d3.scaleLinear()
                     .domain(d3.extent(tsneData, d => d.y))
                     .range([svgHeight - 20, 20]);
  
    // Function to generate pie chart SVG group for each point
    function generatePieChart(ratios, cx, cy) {
      const pie = d3.pie().value(d => d)(ratios);
      const arc = d3.arc().innerRadius(0).outerRadius(7);
  
      // Create a group element to position the pie chart
      const pieGroup = svg.append("g")
                          .attr("transform", `translate(${cx}, ${cy})`);
  
      pieGroup.selectAll('path')
        .data(pie)
        .enter()
        .append('path')
        .attr('d', arc)
        .attr('fill', (d, i) => colorScale(i));
  
      return pieGroup;
    }
  
    // Render the scatter plot and pie charts
    tsneData.forEach((point) => {
      const cx = xScale(point.x);
      const cy = yScale(point.y);
  
      // Add scatter plot points
      svg.append("circle")
         .attr("cx", cx)
         .attr("cy", cy)
         .attr("r", 5)
         .attr("fill", "#000");
  
      // Generate and position pie charts at the same coordinates
      generatePieChart(point.ratios, cx, cy);
    });
  
    // Render the color legend
    renderColorLegend(colorScale, numCategories, containerId);
  }
  
  // Function to render the color legend inside the same container
  function renderColorLegend(colorScale, numCategories, containerId) {
    // Create the color legend container (below the plot)
    const legendHeight = 40;
    const legendContainer = d3.select(containerId)
                             .append("svg")
                             .attr("width", 1000)
                             .attr("height", legendHeight)
                             .attr("viewBox", "0 0 1000 40")
                             .style("margin-top", "20px"); // Optional: adds some space between plot and legend
  
    const legendItems = colorScale.domain();
  
    const legend = legendContainer.append("g")
                                  .attr("transform", "translate(10, 10)");
  
    // Add legend items (rectangles and labels)
    legend.selectAll(".legend-item")
          .data(legendItems)
          .enter()
          .append("g")
          .attr("class", "legend-item")
          .attr("transform", (d, i) => `translate(${i * 60}, 0)`)
          .each(function(d, i) {
            const item = d3.select(this);
  
            // Create rectangle for color
            item.append("rect")
                .attr("width", 20)
                .attr("height", 20)
                .attr("fill", colorScale(d));
  
            // Add text label next to the rectangle
            item.append("text")
                .attr("x", 30)
                .attr("y", 15)
                .text(`${d}`);
          });
  }
  