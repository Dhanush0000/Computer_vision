document.addEventListener('DOMContentLoaded', () => {
    const navButtons = document.querySelectorAll('.nav-button');
    const mainContent = document.querySelector('.content');

    // Function to highlight the active navigation button
    function setActiveButton(button) {
        navButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
    }

    // Load content for Home
    function loadHomeContent() {
        mainContent.innerHTML = `
            <h2>Welcome to Flipkart Smart Vision</h2>
            <p>Scan products and manage inventory with ease using the power of smart vision technology.</p>

            <section>
                <h3>Scan Options</h3>
                <form action="/scan_fruits_veggies" method="post">
                    <button type="submit" class="web-button">Scan Fruits/Vegetables</button>
                </form>
                <form action="/scan_packaged_products" method="post">
                    <button type="submit" class="web-button">Scan Packaged Products</button>
                </form>
            </section>
            <section>
                <h3>Reports</h3>
                <form action="/download_report" method="get">
                    <button type="submit" class="web-button">Generate Report</button>
                </form>
            </section>
        `;
    }

    // Load content for Scan
    function loadScanContent() {
        mainContent.innerHTML = `
            <h2>Scan</h2>
            <p>Use this feature to scan fresh produce or packaged products using advanced image recognition technology.</p>
            <form action="/scan_fruits_veggies" method="post">
                <button type="submit" class="web-button">Scan Fresh Produce</button>
            </form>
            <form action="/scan_packaged_products" method="post">
                <button type="submit" class="web-button">Scan Packaged Products</button>
            </form>
        `;
    }

    // Load content for Inventory
    function loadInventoryContent() {
        mainContent.innerHTML = `
            <h2>Inventory</h2>
            <p>Track your products in real-time and manage stock efficiently.</p>
            <button class="web-button">Check Inventory</button>
        `;
    }

    // Load content for Reports
    function loadReportsContent() {
        mainContent.innerHTML = `
            <h2>Reports</h2>
            <form action="/download_report" method="get">
                <button type="submit" class="web-button">Generate Report</button>
            </form>
        `;
    }

    // Load content for AI-Powered Analytics
    function loadAIContent() {
        mainContent.innerHTML = `
            <h2>AI-Powered Analytics</h2>
            <p>Using advanced AI and machine learning algorithms, you can predict trends, manage inventory more effectively, and gain insights from data-driven reports.</p>
        `;
    }

    // Load content for Settings
    function loadSettingsContent() {
        mainContent.innerHTML = `
            <h2>Settings</h2>
            <p>Configure system settings such as notification preferences and scan intervals.</p>
            <button class="web-button">Change Settings</button>
        `;
    }

    // Attach navigation button click events
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            setActiveButton(button);
            switch (button.id) {
                case 'homeBtn':
                    loadHomeContent();
                    break;
                case 'scanBtn':
                    loadScanContent();
                    break;
                case 'inventoryBtn':
                    loadInventoryContent();
                    break;
                case 'reportsBtn':
                    loadReportsContent();
                    break;
                case 'aiBtn':
                    loadAIContent();
                    break;
                case 'settingsBtn':
                    loadSettingsContent();
                    break;
            }
        });
    });

    // Initially load the home content
    loadHomeContent();
});
