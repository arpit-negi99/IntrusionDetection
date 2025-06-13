# Classroom Intrusion Detection System - Improved Version

## Improvements Made

### 1. Enhanced Login System
- Login protection for all routes (requires authentication)
- Modern, user-friendly login interface
- Better session management
- Logout functionality
- Flash messages for user feedback

### 2. Professional User Interface
- Modern dashboard with statistics
- Real-time camera feed display
- Attendance tracking visualization
- Clean and professional design
- Mobile-responsive layout

### 3. Security Enhancements
- Login required for accessing any part of the system
- Protected routes using Flask decorators
- Improved session management
- Better feedback for users

## How to Use

1. Run the improved Flask application:
   ```
   python app.py
   ```

2. Access the system at http://localhost:5000

3. Login with the following credentials:
   - Username: teacher
   - Password: classroom2024

4. Use the dashboard to:
   - Start/stop the camera
   - Monitor student attendance
   - Add new students
   - View security status

## File Structure

- `app.py` - Main Flask application with improved security
- `templates/` - Directory containing HTML templates
  - `base.html` - Base template with common styling
  - `login.html` - Login page
  - `dashboard.html` - Main dashboard interface
  - `add_student.html` - Add student page
- `utils.py` - Utility functions for face recognition
- `EmailAlert2.py` - Email alert functionality
- `train.py` - Script to train the face recognition model

## Dependencies

See requirements.txt for all required packages.
