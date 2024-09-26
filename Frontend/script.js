document.addEventListener('DOMContentLoaded', () => {
    const navButtons = document.querySelectorAll('.nav-button');
    const mainContent = document.querySelector('.content');

    function setActiveButton(button) {
        navButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
    }

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

    function loadHomeContent() {
        mainContent.innerHTML = `
            <h2>Welcome to Flipkart Smart Vision</h2>
            <p>Scan products and manage inventory with ease using the power of smart vision technology.</p>

            <!-- Overview Section -->
            <section class="overview">
                <h3>What is Flipkart Smart Vision?</h3>
                <p>Flipkart Smart Vision is a cutting-edge solution designed to streamline product scanning, inventory management, and report generation. With quick and accurate scans, real-time data updates, and detailed reports, it's built to improve your efficiency.</p>
            </section>

            <!-- Feature Cards Section -->
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
            </section>

            <div class="scan-options">
                <button id="scanFruitsBtn">Scan Fruits/Vegetables</button>
                <button id="scanProductsBtn">Scan Products</button>
            </div>
        `;
        attachScanListeners(); // Attach listeners after dynamic content is loaded
    }

    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            setActiveButton(button);
            switch (button.id) {
                case 'homeBtn':
                    loadHomeContent();
                    break;
                case 'scanBtn':
                    mainContent.innerHTML = '<p>Scanning page...</p>';
                    break;
                case 'inventoryBtn':
                    mainContent.innerHTML = '<p>Inventory page...</p>';
                    break;
                case 'reportsBtn':
                    mainContent.innerHTML = '<p>Reports page...</p>';
                    break;
                case 'settingsBtn':
                    mainContent.innerHTML = '<p>Settings page...</p>';
                    break;
            }
        });
    });

    // Initially load the home content
    loadHomeContent();
});
