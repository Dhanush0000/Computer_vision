document.addEventListener('DOMContentLoaded', () => {
    const navButtons = document.querySelectorAll('.nav-button');
    const mainContent = document.querySelector('.content');

    // Get all feature cards
    const featureCards = document.querySelectorAll('.feature-card');

    // Function to highlight the active navigation button
    function setActiveButton(button) {
        navButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
    }

    // Load content for each section
    function loadHomeContent() {
        mainContent.innerHTML = `
            <h2>Welcome to Flipkart Smart Vision</h2>
            <p>Scan products and manage inventory with ease using the power of smart vision technology.</p>

            <section class="overview">
                <h3>What is Flipkart Smart Vision?</h3>
                <p>Flipkart Smart Vision is a cutting-edge solution designed to streamline product scanning, inventory management, and report generation. With quick and accurate scans, real-time data updates, and detailed reports, it's built to improve your efficiency.</p>
            </section>

            <section class="features">
                <div class="feature-card">
                    <h4>Quick Scan</h4>
                    <p>Scan fruits, vegetables, and products effortlessly with advanced image recognition.</p>
                </div>
                <div class="feature-card">
                    <h4>Real-Time Inventory</h4>
                    <p>Track and manage your inventory in real time with automated updates.</p>
                </div>
                <div class="feature-card">
                    <h4>Generate Reports</h4>
                    <p>Get detailed reports about products and inventory with just a few clicks.</p>
                </div>
                <div class="feature-card">
                    <h4>AI-Powered Analytics</h4>
                    <p>Use AI to analyze and predict trends in inventory and sales.</p>
                </div>
            </section>

            <div class="scan-options">
                <button id="scanFruitsBtn" class="web-button">Scan Fruits/Vegetables</button>
                <button id="scanProductsBtn" class="web-button">Scan Products</button>
            </div>
        `;
        attachScanListeners(); // Attach listeners for dynamic content
    }

    function loadScanContent() {
        mainContent.innerHTML = `
            <h2>Scan</h2>
            <p>Use this feature to scan fresh produce or packaged products using advanced image recognition technology.</p>
            <div class="scan-options">
                <button class="web-button">Scan Fresh Produce</button>
                <button class="web-button">Scan Packaged Products</button>
            </div>
        `;
    }

    function loadInventoryContent() {
        mainContent.innerHTML = `
            <h2>Inventory</h2>
            <p>Track your products in real-time and manage stock efficiently.</p>
            <button class="web-button">Check Inventory</button>
        `;
    }

    function loadReportsContent() {
        mainContent.innerHTML = `
            <h2>Reports</h2>
            <p>Generate custom reports on your products, sales, and inventory trends.</p>
            <button class="web-button">Generate Report</button>
        `;
    }

    function loadAIContent() {
        mainContent.innerHTML = `
            <h2>AI-Powered Analytics</h2>
            <p>Using advanced AI and machine learning algorithms, you can predict trends, manage inventory more effectively, and gain insights from data-driven reports.</p>
        `;
    }

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

    // Attach click events to the feature cards
    featureCards.forEach((card, index) => {
        card.addEventListener('click', () => {
            setActiveButton(navButtons[index + 1]); // Match feature card to the corresponding nav button
            switch (index) {
                case 0:
                    loadScanContent();
                    break;
                case 1:
                    loadInventoryContent();
                    break;
                case 2:
                    loadReportsContent();
                    break;
                case 3:
                    loadAIContent();
                    break;
            }
        });
    });

    // Function to handle scan button click listeners
    function attachScanListeners() {
        const scanFruitsBtn = document.getElementById('scanFruitsBtn');
        const scanProductsBtn = document.getElementById('scanProductsBtn');

        if (scanFruitsBtn) {
            scanFruitsBtn.addEventListener('click', () => {
                mainContent.innerHTML = `<p>Fruit/Vegetable scanning initiated...</p>`;
            });
        }

        if (scanProductsBtn) {
            scanProductsBtn.addEventListener('click', () => {
                mainContent.innerHTML = `<p>Product scanning initiated...</p>`;
            });
        }
    }

    // Load the home content initially
    loadHomeContent();
});
