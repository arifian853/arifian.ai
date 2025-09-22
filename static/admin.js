// Global variables
let editId = null;

// Load knowledge items
async function loadKnowledge() {
    try {
        const response = await fetch('/knowledge');
        const data = await response.json();

        const knowledgeList = document.getElementById('knowledge-list');
        knowledgeList.innerHTML = '';

        if (data.length === 0) {
            knowledgeList.innerHTML = '<div class="alert alert-info">No knowledge items found.</div>';
            return;
        }

        data.forEach(item => {
            const div = document.createElement('div');
            div.className = 'knowledge-item';
            div.innerHTML = `
                <h4>${item.title}</h4>
                <p><strong>Source:</strong> ${item.source || 'N/A'}</p>
                <p>${item.content.substring(0, 200)}${item.content.length > 200 ? '...' : ''}</p>
                <div class="mt-2">
                    <button class="btn btn-sm btn-primary edit-btn me-2" data-id="${item._id}">Edit</button>
                    <button class="btn btn-sm btn-danger delete-btn" data-id="${item._id}">Delete</button>
                </div>
            `;
            knowledgeList.appendChild(div);
        });

        // Add event listeners to edit buttons
        document.querySelectorAll('.edit-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const id = e.target.getAttribute('data-id');
                try {
                    const response = await fetch(`/knowledge/${id}`);
                    if (response.ok) {
                        const knowledge = await response.json();

                        // Fill form with existing data
                        document.getElementById('title').value = knowledge.title;
                        document.getElementById('content').value = knowledge.content;
                        document.getElementById('source').value = knowledge.source || '';

                        // Store edit ID
                        editId = id;
                        document.getElementById('add-knowledge-form').setAttribute('data-edit-id', id);

                        // Change button text and style
                        const submitBtn = document.querySelector('#add-knowledge-form button[type="submit"]');
                        submitBtn.textContent = 'Update Knowledge';
                        submitBtn.className = 'btn btn-warning';

                        // Switch to add tab
                        document.getElementById('add-tab').click();
                    } else {
                        const error = await response.json();
                        alert(`Error: ${error.detail}`);
                    }
                } catch (error) {
                    alert(`Error: ${error.message}`);
                }
            });
        });

        // Add event listeners to delete buttons
        document.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                if (confirm('Are you sure you want to delete this item?')) {
                    const id = e.target.getAttribute('data-id');
                    try {
                        const response = await fetch(`/knowledge/${id}`, {
                            method: 'DELETE'
                        });

                        if (response.ok) {
                            alert('Knowledge deleted successfully');
                            loadKnowledge();
                        } else {
                            const error = await response.json();
                            alert(`Error: ${error.detail}`);
                        }
                    } catch (error) {
                        alert(`Error: ${error.message}`);
                    }
                }
            });
        });
    } catch (error) {
        console.error('Error loading knowledge:', error);
        document.getElementById('knowledge-list').innerHTML = `
            <div class="alert alert-danger">Error loading knowledge: ${error.message}</div>
        `;
    }
}

// Add/Update knowledge form
function setupKnowledgeForm() {
    document.getElementById('add-knowledge-form').addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = {
            title: document.getElementById('title').value,
            content: document.getElementById('content').value,
            source: document.getElementById('source').value || null
        };

        const isEditing = editId !== null;
        const url = isEditing ? `/knowledge/${editId}` : '/knowledge';
        const method = isEditing ? 'PUT' : 'POST';

        try {
            const response = await fetch(url, {
                method: method,
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            if (response.ok) {
                const message = isEditing ? 'Knowledge updated successfully' : 'Knowledge added successfully';
                alert(message);

                // Reset form and edit state
                document.getElementById('add-knowledge-form').reset();
                document.getElementById('add-knowledge-form').removeAttribute('data-edit-id');
                editId = null;

                // Reset button
                const submitBtn = document.querySelector('#add-knowledge-form button[type="submit"]');
                submitBtn.textContent = 'Add Knowledge';
                submitBtn.className = 'btn btn-primary';

                loadKnowledge();
                // Switch to knowledge tab
                document.getElementById('knowledge-tab').click();
            } else {
                const error = await response.json();
                alert(`Error: ${error.detail}`);
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    });
}

// Upload TXT form
function setupTxtUpload() {
    document.getElementById('upload-txt-form').addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData();
        formData.append('file', document.getElementById('txt-file').files[0]);
        formData.append('title', document.getElementById('txt-title').value);
        formData.append('source', document.getElementById('txt-source').value || '');

        try {
            const response = await fetch('/upload-txt', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                alert('Text file uploaded successfully');
                document.getElementById('upload-txt-form').reset();
                loadKnowledge();
            } else {
                const error = await response.json();
                alert(`Error: ${error.detail}`);
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    });
}

// Upload CSV form
function setupCsvUpload() {
    document.getElementById('upload-csv-form').addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData();
        formData.append('file', document.getElementById('csv-file').files[0]);
        formData.append('title_column', document.getElementById('title-column').value);
        formData.append('content_column', document.getElementById('content-column').value);

        try {
            const response = await fetch('/upload-csv', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                alert(`CSV file processed: ${result.message}`);
                document.getElementById('upload-csv-form').reset();
                loadKnowledge();
            } else {
                const error = await response.json();
                alert(`Error: ${error.detail}`);
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    });
}

// Upload JSON form
function setupJsonUpload() {
    document.getElementById('upload-json-form').addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData();
        formData.append('file', document.getElementById('json-file').files[0]);
        formData.append('title_field', document.getElementById('title-field').value);
        formData.append('content_field', document.getElementById('content-field').value);

        try {
            const response = await fetch('/upload-json', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                alert(`JSON file processed: ${result.message}`);
                document.getElementById('upload-json-form').reset();
                loadKnowledge();
            } else {
                const error = await response.json();
                alert(`Error: ${error.detail}`);
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    });
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    loadKnowledge();
    setupKnowledgeForm();
    setupTxtUpload();
    setupCsvUpload();
    setupJsonUpload();
});