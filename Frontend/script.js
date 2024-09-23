document.addEventListener('DOMContentLoaded', () => {
    const navButtons = document.querySelectorAll('.nav-button');
    const mainContent = document.querySelector('.content');

    function setActiveButton(button) {
        navButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
    }

    function loadHomeContent() {
        mainContent.innerHTML = `
            <h2>Welcome to Flipkart Smart Vision</h2>
            <p>Scan products and fruits/vegetables quickly and efficiently.</p>
            <div class="scan-options">
                <button id="scanFruitsBtn">Scan Fruits/Vegetables</button>
                <button id="scanProductsBtn">Scan Products</button>
            </div>
        `;
    }

    function loadScanContent() {
        mainContent.innerHTML = `
            <h3>Fruits/Vegetables Scanning</h3>
            <p>Scan fresh produce items.</p>
            <button id="startFruitsScan">Start Scanning</button>

            <h3>Product Scanning</h3>
            <p>Scan barcodes or OCR products.</p>
            <button id="startProductsScan">Start Scanning</button>
        `;
    }

    function loadInventoryContent() {
        const inventoryContent = `
            <table>
                <thead>
                    <tr>
                        <th>Product</th>
                        <th>Quantity</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Inventory items will be dynamically added -->
                </tbody>
            </table>
        `;
        mainContent.innerHTML = inventoryContent;
    }

    function loadReportsContent() {
        const reportsContent = `
            <h2>Generate Reports</h2>
            <form id="reportForm">
                <select name="reportType" id="reportType">
                    <option value="">Select Report Type</option>
                    <option value="setOfProducts">Set of Products</option>
                    <option value="last10Products">Last 10 Products</option>
                    <option value="todaysReport">Today's Report</option>
                </select>
                <button type="submit">Generate Report</button>
            </form>
        `;
        mainContent.innerHTML = reportsContent;
    }

    function loadSettingsContent() {
        const settingsContent = `
            <h2>System Settings</h2>
            <form id="settingsForm">
                <label for="notificationFrequency">Notification Frequency:</label>
                <input type="number" id="notificationFrequency" min="1" max="100">
                <button type="submit">Save Settings</button>
            </form>
        `;
        mainContent.innerHTML = settingsContent;
    }

    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            setActiveButton(button);
            switch (button.id) {
                case 'home':
                    loadHomeContent();
                    break;
                case 'scan':
                    loadScanContent();
                    break;
                case 'inventory':
                    loadInventoryContent();
                    break;
                case 'reports':
                    loadReportsContent();
                    break;
                case 'settings':
                    loadSettingsContent();
                    break;
            }
        });
    });

    // Simulating scan functionality
    const scanOptions = document.querySelectorAll('.scan-option');
    let scannedItems = 0;

    scanOptions.forEach(option => {
        option.querySelector('#scanProductsBtn').addEventListener('click', () => {
            console.log('Scanning products...');
            scannedItems++;
            alert(`Product scanned. ${scannedItems} items scanned.`);
            if (scannedItems % 10 === 0) {
                alert('Notification: 10 items scanned!');
            }
        });

        option.querySelector('#scanFruitsBtn').addEventListener('click', () => {
            console.log('Scanning fruits/vegetables...');
            scannedItems++;
            alert(`Fresh produce item scanned. ${scannedItems} items scanned.`);
            if (scannedItems % 10 === 0) {
                alert('Notification: 10 items scanned!');
            }
        });
    });
});
