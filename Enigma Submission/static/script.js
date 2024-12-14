document.getElementById('stockForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const symbol = document.getElementById('symbol').value;
    const period = document.getElementById('period').value;

    fetch('/fetch_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `symbol=${symbol}&period=${period}`
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('message').innerText = data.error;
        } else {
            document.getElementById('message').innerText = data.message;
            const stockData = data.data;
            let tableHtml = `<table>
                                <tr>
                                    <th>Date</th>
                                    <th>Close Price</th>
                                </tr>`;
            for (let i = 0; i < stockData['Date'].length; i++) {
                tableHtml += `<tr>
                                <td>${stockData['Date'][i]}</td>
                                <td>${stockData['Close'][i]}</td>
                              </tr>`;
            }
            tableHtml += `</table>`;
            document.getElementById('stockTable').innerHTML = tableHtml;
        }
    })
    .catch(err => console.error('Error:', err));
});
