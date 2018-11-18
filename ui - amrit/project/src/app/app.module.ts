import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { MainComponent } from './main/main.component';
import { LandingComponent } from './landing/landing.component';
import { FileDropDirective } from './file-drop.directive';

@NgModule({
  declarations: [
    AppComponent,
    MainComponent,
    LandingComponent,
    FileDropDirective
  ],
  imports: [
    BrowserModule,
    AppRoutingModule
  ],
  providers: [
    
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
