function placeOrder() {
    let item = document.getElementById("item").value;

    fetch('/get_shelf_location', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ item: item })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("orderStatus").innerText = data.message || data.error;
    })
    .catch(error => console.error('Error:', error));
}

function requestRobot() {
    let shelf = document.getElementById("shelf").value;
    let command = document.getElementById("command").value;

    if (!command) {
        document.getElementById("assistStatus").innerText = "Please enter a command.";
        return;
    }

    fetch('/get_shelf_location', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ item: shelf })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("assistStatus").innerText = data.message || data.error;
    })
    .catch(error => console.error('Error:', error));
}
