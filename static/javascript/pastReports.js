document.addEventListener("DOMContentLoaded", () => {
  const root = document.getElementById("past-reports-root");
  if (!root) return;

  const userId = root.getAttribute("data-user-id") || null;

  // Loading state
  root.innerHTML = `<div class="text-center p-4 text-muted">Loading reports...</div>`;

  if (!userId) {
    root.innerHTML = `<div class="text-danger p-3">User not found</div>`;
    return;
  }

  fetch(`/api/history/${userId}`)
    .then((res) => {
      if (!res.ok) throw new Error("Failed to load reports");
      return res.json();
    })
    .then((data) => {
      const reports = (data.reports || []).slice(0, 5);

      // No reports
      if (reports.length === 0) {
        root.innerHTML = `
          <div class="text-center p-4 text-muted">
            <p>No reports found</p>
          </div>
        `;
        return;
      }

      // Build table
      let tableHTML = `
        <div class="table-responsive">
          <table class="table table-striped">
            <thead>
              <tr>
                <th>#</th>
                <th>Disease</th>
                <th>Date</th>
                <th>Severity</th>
              </tr>
            </thead>
            <tbody>
      `;

      reports.forEach((r, idx) => {
        let badge = "";

        if (r.severity === "High") {
          badge = `<span class="badge bg-danger">High</span>`;
        } else if (r.severity === "Medium") {
          badge = `<span class="badge bg-warning text-dark">Medium</span>`;
        } else if (r.severity === "Low") {
          badge = `<span class="badge bg-success">Low</span>`;
        } else {
          badge = `<span class="badge bg-secondary">${r.severity}</span>`;
        }

        tableHTML += `
          <tr>
            <td>${idx + 1}</td>
            <td><strong>${r.disease}</strong></td>
            <td>${new Date(r.date).toLocaleString()}</td>
            <td>${badge}</td>
          </tr>
        `;
      });

      tableHTML += `
            </tbody>
          </table>
        </div>
      `;

      root.innerHTML = tableHTML;
    })
    .catch((err) => {
      root.innerHTML = `
        <div class="text-danger p-3">
          Error loading reports: ${err.message}
        </div>
      `;
    });
});