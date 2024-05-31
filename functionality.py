from PyQt5 import QtCore, QtGui, QtWidgets
import tranformertechnique
from glovetechnique import lexrank_summary, load_glove_embeddings
import summary
class functionality(QtWidgets.QMainWindow,summary.Ui_MainWindow): 
    def __init__(self): 
        super().__init__()
        self.setupUi(self)
        self.setup_connections()
        self.add_items_combo()
    def add_items_combo(self): 
        self.comboBox.addItem('Transformer')
        self.comboBox.addItem('GloVe')
    def setup_connections(self): 
        self.pushButton.clicked.connect(self.generate_summary)
    def generate_summary(self): 
        try:
            input_text = self.textEdit.toPlainText()
            selected_model = self.comboBox.currentText()

            if selected_model == "Transformer":
                summary_text = tranformertechnique.generate_transformer_summary(input_text)
            elif selected_model == "GloVe":
                embeddings_index = load_glove_embeddings('Application//glove.6B.100d.txt')
                summary_text = lexrank_summary(input_text, embeddings_index)

            self.textEdit_2.setPlainText(summary_text)
            self.statusbar.showMessage("Summarization complete")
        except FileNotFoundError:
            error_message = "The GloVe embeddings file 'glove.6B.100d.txt' was not found."
            self.textEdit_2.setPlainText(error_message)
            self.statusbar.showMessage("Summarization failed: File not found")
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            self.textEdit_2.setPlainText(error_message)
            self.statusbar.showMessage("Summarization failed")
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = functionality()
    window.show()
    sys.exit(app.exec_())