document.addEventListener('DOMContentLoaded', () => {
    const navButtons = document.querySelectorAll('.nav-button');
    const mainContent = document.querySelector('.content');
    const logoLink = document.getElementById('logoLink');

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

    // Attach event listener to logo to redirect to home
    logoLink.addEventListener('click', (e) => {
        e.preventDefault();
        loadHomeContent();
        setActiveButton(document.getElementById('homeBtn')); // Set home button as active
    });

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
