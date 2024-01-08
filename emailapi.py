import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QListWidget, QTextBrowser, QPushButton, QWidget, QMessageBox
from PyQt5.QtCore import Qt
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/gmail.labels"]

class EmailViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Email Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.label_list = QListWidget(self)
        self.label_list.setSelectionMode(QListWidget.SingleSelection)
        self.label_list.itemClicked.connect(self.display_label_info)

        self.label_info = QTextBrowser(self)

        self.refresh_button = QPushButton("Refresh Labels", self)
        self.refresh_button.clicked.connect(self.refresh_labels)

        self.delete_label_button = QPushButton("Delete Label", self)
        self.delete_label_button.clicked.connect(self.delete_selected_label)

        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.label_list)
        layout.addWidget(self.label_info)
        layout.addWidget(self.delete_label_button)

        self.credentials = self.get_credentials()
        self.service = build("gmail", "v1", credentials=self.credentials)

        self.refresh_labels()

    def get_credentials(self):
        creds = None

        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("emailt.json", SCOPES)
                creds = flow.run_local_server(port=0)

            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return creds

    def refresh_labels(self):
        self.label_list.clear()
        try:
            labels = self.service.users().labels().list(userId="me").execute().get('labels', [])

            if not labels:
                print("No labels found.")
                return

            for label in labels:
                self.label_list.addItem(label['name'])

        except HttpError as error:
            print(f"An error occurred: {error}")

    def display_label_info(self, item):
        label_name = item.text()
        self.label_info.setPlainText(f"Label Name: {label_name}")

    def delete_selected_label(self):
        selected_item = self.label_list.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "No Label Selected", "Please select a label to delete.")
            return

        confirmation = QMessageBox.question(self, "Confirm Deletion", "Are you sure you want to delete the selected label?",
                                           QMessageBox.Yes | QMessageBox.No)
        if confirmation == QMessageBox.Yes:
            label_name = selected_item.text()
            try:
                self.service.users().labels().delete(userId="me", id=label_name).execute()
                self.refresh_labels()
            except HttpError as error:
                print(f"Error deleting label: {error}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    email_viewer = EmailViewerApp()
    email_viewer.show()
    sys.exit(app.exec_())
