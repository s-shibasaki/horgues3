
namespace JVDataCollector
{
    partial class JVLinkForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(JVLinkForm));
            this.axJVLink1 = new AxJVDTLabLib.AxJVLink();
            ((System.ComponentModel.ISupportInitialize)(this.axJVLink1)).BeginInit();
            this.SuspendLayout();
            // 
            // axJVLink1
            // 
            this.axJVLink1.Enabled = true;
            this.axJVLink1.Location = new System.Drawing.Point(0, 0);
            this.axJVLink1.Name = "axJVLink1";
            this.axJVLink1.OcxState = ((System.Windows.Forms.AxHost.State)(resources.GetObject("axJVLink1.OcxState")));
            this.axJVLink1.Size = new System.Drawing.Size(288, 288);
            this.axJVLink1.TabIndex = 0;
            // 
            // JVLinkForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.axJVLink1);
            this.Name = "JVLinkForm";
            this.Text = "JVLinkForm";
            this.Load += new System.EventHandler(this.JVLinkForm_Load);
            ((System.ComponentModel.ISupportInitialize)(this.axJVLink1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private AxJVDTLabLib.AxJVLink axJVLink1;
    }
}